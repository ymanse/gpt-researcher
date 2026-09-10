"""The embedding router decides the easy queries and declines the rest.

`_classify_query` used to spend one FAST_LLM call -- one `claude` CLI session on the
subscription -- to pick which retriever bundle a query is searched with. The choice is
between five fixed categories, which is a nearest-neighbour problem, not a reasoning
one, and the local encoder that already serves the merge stage answers it for free.

What these tests pin is not the accuracy (that is measured, not asserted -- see the
live test at the bottom) but the SAFETY PROPERTY that makes the swap free:

    the router either decides confidently or hands the query to the LLM unchanged.

Every way the router can fail -- no encoder configured, encoder down, encoder unsure --
has to land on the old path. If any of them can instead return a category, the run gets
a bundle nobody chose and the failure is invisible: a wrongly-routed search still
returns results, just worse ones.

The no-encoder case is also what keeps the s11 contract
(`test_s11_classify_runs_once_per_run`) meaningful: those configs carry no
`embedding_provider`, so they still measure the LLM path exactly as before.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from gpt_researcher.retrievers.smart import smart_retriever as sr
from gpt_researcher.retrievers.smart.smart_retriever import (
    CATEGORY_PROTOTYPES, ROUTING_TABLE, SmartRetriever)
from gpt_researcher.utils import llm as llm_module

from tests.search_quality.s12.classification_set import LABELLED_QUERIES

EMBED_CFG = dict(embedding_provider="openai", embedding_model="qwen3-embedding-4b",
                 embedding_kwargs={"openai_api_base": "http://localhost:1236/v1",
                                   "openai_api_key": "local",
                                   "check_embedding_ctx_length": False})
_CACHE_KEY = ("openai", "qwen3-embedding-4b")


def _cfg(**over):
    """The config shape the classifier sees. Mirrors s11's helper, plus embeddings."""
    base = dict(fast_llm_provider="fake", fast_llm_model="fake-model",
                llm_kwargs={}, smart_retriever_config=None,
                smart_retriever_force_category=None, config_path=None)
    base.update(over)
    return SimpleNamespace(**base)


class _LLMBoundary:
    """Stands in for the FAST_LLM, and counts whether it was asked at all."""

    def __init__(self, verdict="academic"):
        self.verdict, self.calls = verdict, []

    async def __call__(self, *a, **kw):
        self.calls.append(kw)
        return self.verdict


def _onehot():
    """A centroid per category, each its own axis -- so a query vector equal to one of
    them scores cosine 1.0 against it and 0.0 against every other: margin 1.0, which is
    unambiguously above any threshold the router might ship."""
    return {c: [1.0 if i == n else 0.0 for i in range(len(ROUTING_TABLE))]
            for n, c in enumerate(sorted(ROUTING_TABLE))}


def _install(monkeypatch, embedder, centroids):
    """Install a deterministic encoder, bypassing the network and the prototypes.

    Keyed into the cache directly rather than through `Memory`: the seam under test is
    what `_classify_query` does with a router's verdict, not how the client is built.
    """
    monkeypatch.setitem(sr._CENTROID_CACHE, _CACHE_KEY, (embedder, centroids))


# --------------------------------------------------------- the router decides

def test_a_confident_encoder_routes_without_spending_an_llm_call(monkeypatch):
    """The whole point: a clear query costs no CLI session."""
    centroids = _onehot()
    target = sorted(ROUTING_TABLE)[2]

    class _Fake:
        def embed_query(self, text):
            return centroids[target]

    _install(monkeypatch, _Fake(), centroids)

    boundary = _LLMBoundary()
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("q", cfg=_cfg(**EMBED_CFG))._classify_query()

    assert category == target, f"the encoder's verdict must be used, got {category!r}"
    assert boundary.calls == [], (
        f"{len(boundary.calls)} FAST_LLM calls although the encoder was confident -- "
        "the routing decision is the session this router exists to stop spending")


# --------------------------------------------------------- the router declines

def test_an_unsure_encoder_hands_the_query_to_the_llm(monkeypatch):
    """Two categories within the margin is exactly when a guess is worth least.

    Every centroid is the same vector here, so every cosine ties and the margin is 0 --
    the most unsure the router can possibly be.
    """
    tied = {c: [1.0, 1.0, 0.0, 0.0, 0.0] for c in sorted(ROUTING_TABLE)}

    class _Fake:
        def embed_query(self, text):
            return [1.0, 1.0, 0.0, 0.0, 0.0]

    _install(monkeypatch, _Fake(), tied)

    boundary = _LLMBoundary(verdict="news_current")
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("q", cfg=_cfg(**EMBED_CFG))._classify_query()

    assert len(boundary.calls) == 1, (
        f"{len(boundary.calls)} FAST_LLM calls for a query the encoder could not "
        "separate -- an unsure router must fall through, not guess a bundle")
    assert category == "news_current", (
        f"the LLM's verdict must win once it is asked, got {category!r}")


def test_a_dead_encoder_falls_back_instead_of_failing_the_search(monkeypatch):
    """The embedding server is a separate process; it being down must cost accuracy,
    never the query. This is the case that would otherwise turn a routing optimisation
    into an outage."""
    class _Broken:
        def embed_query(self, text):
            raise ConnectionError("embedding server refused the connection")

    _install(monkeypatch, _Broken(), _onehot())

    boundary = _LLMBoundary(verdict="code_technical")
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("q", cfg=_cfg(**EMBED_CFG))._classify_query()

    assert len(boundary.calls) == 1, (
        "a dead encoder must degrade to the FAST_LLM, but it was never asked")
    assert category == "code_technical", f"got {category!r}"


def test_no_configured_encoder_leaves_the_old_path_untouched():
    """The s11 contract. Those configs carry no embedding settings, so they must still
    measure the LLM path -- otherwise this change would silently hollow them out."""
    boundary = _LLMBoundary(verdict="academic")
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("q", cfg=_cfg())._classify_query()

    assert len(boundary.calls) == 1, (
        f"{len(boundary.calls)} FAST_LLM calls with no encoder configured -- the "
        "pre-router behaviour has to survive exactly, or s11 passes vacuously")
    assert category == "academic", f"got {category!r}"


def test_a_forced_category_still_beats_the_encoder(monkeypatch):
    """P1.1 stamps the run's category onto every node. That short-circuit sits ABOVE
    the router and must stay there -- re-encoding per node would give back the
    per-node cost the stamping removed."""
    encoded = []

    class _Counting:
        def embed_query(self, text):
            encoded.append(text)
            return [1.0, 0.0, 0.0, 0.0, 0.0]

    _install(monkeypatch, _Counting(), _onehot())

    boundary = _LLMBoundary()
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("q", cfg=_cfg(
            smart_retriever_force_category="academic", **EMBED_CFG))._classify_query()

    assert category == "academic", f"the forced category must win, got {category!r}"
    assert encoded == [] and boundary.calls == [], (
        f"a forced category still cost {len(encoded)} encodes and "
        f"{len(boundary.calls)} LLM calls; it has to cost neither")


def test_a_failing_encoder_is_not_rebuilt_on_every_query(monkeypatch):
    """Building the centroids is 25 prototype embeddings. An unremembered failure
    re-attempts all 25 for every query, forever, IN FRONT OF the LLM call it was meant
    to replace -- which makes a misconfigured encoder strictly worse than none.

    Measured against a real out-of-credit OpenAI endpoint before this cooldown existed:
    three queries, three full failing builds.
    """
    attempts = []

    class _DeadMemory:
        def __init__(self, *a, **kw):
            attempts.append(1)
            raise ConnectionError("no route to the embedding server")

    monkeypatch.setattr(sr, "_CENTROID_CACHE", {})
    monkeypatch.setattr(sr, "_ROUTER_RETRY_AFTER", {})
    monkeypatch.setattr("gpt_researcher.memory.embeddings.Memory", _DeadMemory)

    for _ in range(5):
        assert sr._router(SimpleNamespace(**EMBED_CFG)) == (None, None)

    assert len(attempts) == 1, (
        f"the dead encoder was rebuilt {len(attempts)} times over 5 queries -- a failure "
        "has to be remembered for SMART_RETRIEVER_ROUTER_COOLDOWN_S, not re-paid")


def test_the_cooldown_expires_so_a_blip_does_not_disable_routing_until_restart(monkeypatch):
    """The encoder is a separate service. A tombstone that never expires would turn one
    bad second into LLM routing for the life of the MCP process."""
    attempts = []

    class _DeadMemory:
        def __init__(self, *a, **kw):
            attempts.append(1)
            raise ConnectionError("blip")

    monkeypatch.setattr(sr, "_CENTROID_CACHE", {})
    monkeypatch.setattr(sr, "_ROUTER_RETRY_AFTER", {})
    monkeypatch.setattr("gpt_researcher.memory.embeddings.Memory", _DeadMemory)

    sr._router(SimpleNamespace(**EMBED_CFG))
    assert len(attempts) == 1

    # walk past the deadline rather than sleeping through it
    monkeypatch.setattr(sr.time, "monotonic",
                        lambda: sr._ROUTER_RETRY_AFTER[_CACHE_KEY] + 1.0)
    sr._router(SimpleNamespace(**EMBED_CFG))

    assert len(attempts) == 2, (
        "the encoder was never retried after the cooldown expired -- one blip would "
        "disable routing until the process restarts")


# ------------------------------------------------------------- the prototypes

def test_every_category_has_prototypes_and_they_are_all_routable():
    """A prototype for a category ROUTING_TABLE has no bundle for would build a
    centroid the router can pick and then cannot serve."""
    assert set(CATEGORY_PROTOTYPES) == set(ROUTING_TABLE), (
        f"prototype categories {sorted(CATEGORY_PROTOTYPES)} do not match routable "
        f"categories {sorted(ROUTING_TABLE)}")
    thin = {c: len(p) for c, p in CATEGORY_PROTOTYPES.items() if len(p) < 3}
    assert not thin, (
        f"categories with too few prototypes {thin} -- a centroid over one or two "
        "queries is a point on one topic, and every off-topic query loses to a "
        "category whose prototypes happen to be broader")


# -------------------------------------------------------------- live accuracy

def test_the_encoder_never_routes_a_labelled_query_to_the_wrong_bundle():
    """The measurement, against the hand-labelled set. Skips when the encoder is not
    running -- it is a separate service, and CI without it must not go red.

    Asserts the SAFETY property, not an accuracy target: of the queries the router
    decides at the shipped margin, none may be wrong. Coverage is reported rather than
    asserted -- it is the dial (`SMART_RETRIEVER_EMBED_MARGIN`) that trades sessions
    for confidence, and the sweep measured 2026-09-10 on this set was:

        margin  coverage  wrong          margin  coverage  wrong
          0.00      100%      2            0.03       70%      0
          0.01       98%      2            0.05       65%      0   <- shipped
          0.02       88%      1            0.08       45%      0

    0.05 rather than 0.03 for the same clean result: on 40 queries, buying 5% coverage
    by halving the distance to the 0.02 cliff is the wrong side of that trade.
    """
    sr._CENTROID_CACHE.pop(_CACHE_KEY, None)      # never score against a faked router
    try:
        embedder, centroids = sr._router(SimpleNamespace(**EMBED_CFG))
        assert embedder is not None and centroids and embedder.embed_query("probe")
    except Exception as exc:
        pytest.skip(f"embedding server unavailable at localhost:1236 ({exc})")

    wrong, decided = [], 0
    for query, want in LABELLED_QUERIES:
        vector = embedder.embed_query(query)
        ranked = sorted(((sr._cos(vector, c), cat) for cat, c in centroids.items()),
                        reverse=True)
        if ranked[0][0] - ranked[1][0] < sr._EMBED_MARGIN:
            continue                      # declined -> the LLM answers, as before
        decided += 1
        if ranked[0][1] != want:
            wrong.append(f"{query!r}: routed {ranked[0][1]}, labelled {want}")

    print(f"\nrouted {decided}/{len(LABELLED_QUERIES)} "
          f"({decided / len(LABELLED_QUERIES) * 100:.0f}% coverage) at margin "
          f"{sr._EMBED_MARGIN}, {len(wrong)} wrong")
    assert not wrong, (
        f"{len(wrong)} of {decided} routed queries went to the wrong bundle at margin "
        f"{sr._EMBED_MARGIN} -- raise SMART_RETRIEVER_EMBED_MARGIN or fix the "
        "prototypes:\n  " + "\n  ".join(wrong))
