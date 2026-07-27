"""s1 — a retriever failure must be said out loud, at the level the spec assigns it.

Measured in the round-2 live probe: retriever_errors read 0 for a deployment whose
tavily answers 432 to every call. Two holes produced that reading, and both are the
same defect — a failure that leaves no log record at all (a stdout print, or nothing)
is invisible:

  (a) SmartRetriever swallowed each per-retriever exception into an empty list;
  (b) every retriever that catches its own exception printed it to stdout instead of
      logging it — duckduckgo, the keyless head of the fallback order, included.

Which level the record gets is fixed by spec/search-quality.md s1 (2026-07-27):
retriever_errors counts UNRECOVERED failures only. A failure a sibling in the same
parallel bundle — or the fallback route — covered for is a WARNING; the ERROR record
is reserved for a query that ultimately came back with nothing. Counting recovered
failures would make the live gate's `retriever_errors == 0` a verdict on the external
services' mood that day instead of on this code.

Deterministic: no network, no LLM. requests.post is mocked and ddgs is replaced by
an in-memory fake module.
"""
import importlib.machinery
import logging
import sys
import types
from types import SimpleNamespace

import requests

# The literal the live probe (harness-search/scripts/container_probe_s1.py) greps for
# and counts. Spelled out here rather than imported, on purpose: importing it from the
# implementation would let a rename pass both sides silently while marker_wired goes
# false at live-measure time.
MARKER = "SQ_RETRIEVAL_UNRECOVERED"


class _Resp432:
    status_code = 432

    def raise_for_status(self):
        raise requests.HTTPError("432 Client Error: rate limited")

    def json(self):
        return {}


def _tavily_432_route(monkeypatch, other_retriever):
    """A tavily-only route where tavily 432s every call, as measured. `other_retriever`
    serves every OTHER name — i.e. whatever the fallback route reaches for."""
    import gpt_researcher.actions.retriever as retriever_actions
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp432())
    monkeypatch.setattr(
        retriever_actions,
        "get_retriever",
        lambda name: TavilySearch if name == "tavily" else other_retriever,
    )
    return SimpleNamespace(
        smart_retriever_force_category="general_web",
        smart_retriever_config={"general_web": [("tavily", 7, {})]},
    )


def _records(caplog, level):
    return [rec.getMessage() for rec in caplog.records if rec.levelno == level]


def test_tavily_432_a_fallback_covers_for_is_a_warning_not_an_error(monkeypatch, caplog):
    from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever

    fallback_results = [{"href": "https://fallback.example.com/a", "body": "fallback body A"}]

    class _FakeFallbackRetriever:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return fallback_results

    cfg = _tavily_432_route(monkeypatch, _FakeFallbackRetriever)
    with caplog.at_level(logging.DEBUG):
        results = SmartRetriever("some query", cfg=cfg).search(max_results=10)

    assert {r.get("href") for r in results} == {"https://fallback.example.com/a"}, (
        "precondition: the fallback still has to cover for the 432"
    )
    errors = _records(caplog, logging.ERROR)
    assert not errors, (
        "a failure the fallback covered for is a RECOVERED failure — the spec counts "
        "unrecovered ones only, and an ERROR here makes the live gate "
        f"(retriever_errors == 0) fail on a query that actually succeeded. records: {errors}"
    )
    assert any("tavily" in msg for msg in _records(caplog, logging.WARNING)), (
        "recovered is not silent: the 432 must still leave a WARNING naming tavily, "
        f"or a retriever that 432s every call reads as healthy. records: {caplog.messages}"
    )
    assert not [msg for msg in caplog.messages if MARKER in msg], (
        f"{MARKER} marks UNRECOVERED retrieval only. Asserted across every level, not "
        "just ERROR: a covered failure tagged at WARNING today is one severity bump "
        "away from being counted, and then the live gate is back at the mercy of "
        f"external service weather. records: {caplog.messages}"
    )


def test_tavily_432_no_one_covers_for_emits_an_error_naming_it(monkeypatch, caplog):
    from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever

    class _EmptyRetriever:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return []

    cfg = _tavily_432_route(monkeypatch, _EmptyRetriever)
    with caplog.at_level(logging.DEBUG):
        results = SmartRetriever("some query", cfg=cfg).search(max_results=10)

    assert results == [], "precondition: nothing covered for the 432 this time"
    errors = _records(caplog, logging.ERROR)
    assert any("tavily" in msg for msg in errors), (
        "an UNRECOVERED failure must leave an ERROR record naming the retriever — "
        "this is the total-loss case the live probe exists to catch, and it must "
        f"fail the gate. records: {errors}"
    )
    assert len(errors) == 1, (
        f"one query that came back empty is one ERROR, not a pile. got {errors}"
    )
    assert [msg for msg in errors if MARKER in msg] == errors, (
        f"the ERROR has to carry the {MARKER} marker verbatim. The live probe counts "
        "ONLY marker-carrying records (so an adapter's own logger.error for a failure "
        "a sibling covered cannot inflate the count), which means an unmarked ERROR "
        f"here is an unrecovered total loss the gate never sees. records: {errors}"
    )


def test_duckduckgo_failure_is_logged_at_error_not_printed(monkeypatch, caplog, capsys):
    fake_mod = types.ModuleType("ddgs")
    fake_mod.__spec__ = importlib.machinery.ModuleSpec("ddgs", loader=None)

    class _FakeDDGS:
        def text(self, query, region=None, max_results=None, **kwargs):
            raise RuntimeError("ddgs rate limited")

    fake_mod.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)

    from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo

    with caplog.at_level(logging.ERROR):
        results = Duckduckgo("test query").search(max_results=3)

    assert results == []
    errors = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert any("ddgs rate limited" in msg for msg in errors), (
        "duckduckgo heads the fallback order; its failure must reach the log, not "
        f"stdout — the probe cannot count a print(). records: {errors}"
    )
    assert "ddgs rate limited" not in capsys.readouterr().out
