"""s1 - defect: the exa retriever calls the installed exa-py with an argument it dropped.

Measured 2026-09-09 in the container (exa-py 2.20.0, added to the image on 2026-09-06
precisely so this retriever would work):

    Retriever 'exa' failed (Exa.search() got an unexpected keyword argument 'use_autoprompt')
    Retriever 'exa' failed twice - retired from routing for this process

`Exa.search` is keyword-only after the query and its parameter set has moved on:

    (self, query, *, stream, contents, num_results, include_domains, exclude_domains,
     start_crawl_date, end_crawl_date, start_published_date, end_published_date,
     include_text, exclude_text, type, category, flags, moderation, user_location,
     additional_queries, system_prompt, output_schema)

`use_autoprompt` is gone; `type`, `num_results` and `include_domains` - the three things
the retriever actually means - are all still there. So exa dies on every call over one
retired argument, and the code_technical route loses its top retriever for the process.

The contract these tests pin is deliberately version-independent, because the host venv
has NO exa_py at all while the container has 2.20.0: the retriever must send what the
client declares and must not send what it does not, whatever version that client is.
Hence a faked client rather than the installed one.

Deterministic: no network, no real exa_py, no API key.
"""
import sys
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import pytest

from gpt_researcher.retrievers.exa.exa import ExaSearch

QUERY = "how do transactional outbox implementations dedupe delivery?"
DOMAINS = ["github.com", "stackoverflow.com"]

# Separates "the retriever passed None" from "the retriever did not pass it at all",
# which is the whole question these tests ask.
_UNSET = object()


def _response():
    """What exa-py 2.20 search() returns without contents: Result.text stays None."""
    return SimpleNamespace(results=[
        SimpleNamespace(url="https://a.example/one", text=None),
        SimpleNamespace(url="https://b.example/two", text=None),
    ])


class _Exa220:
    """Mimics the installed exa-py 2.20.0: keyword-only, and no `use_autoprompt`."""

    def __init__(self, api_key=None, **_ignored):
        self.sent = None
        self.query = None

    def search(self, query, *, num_results=_UNSET, include_domains=_UNSET,
               exclude_domains=_UNSET, type=_UNSET, category=_UNSET, flags=_UNSET):
        self.query = query
        self.sent = {name: value for name, value in (
            ("num_results", num_results), ("include_domains", include_domains),
            ("exclude_domains", exclude_domains), ("type", type),
            ("category", category), ("flags", flags),
        ) if value is not _UNSET}
        return _response()


class _ExaNarrower(_Exa220):
    """A hypothetical next version that also drops `type`.

    Nothing about 2.20 in particular is the contract; the point is that the retriever
    survives the NEXT parameter removal without another edit, which a hard-coded
    argument list cannot.
    """

    def search(self, query, *, num_results=_UNSET, include_domains=_UNSET):
        self.query = query
        self.sent = {name: value for name, value in (
            ("num_results", num_results), ("include_domains", include_domains),
        ) if value is not _UNSET}
        return _response()


class _ExaVarKeyword(_Exa220):
    """A client that declares **kwargs - the shape `Exa.search_and_contents` already has
    in 2.20.0, `(self, query, **kwargs)`.

    Filtering by declared name against this one strips every argument, so the client
    answers with its own unfiltered defaults: no search mode, no limit, no domains, and
    no error either. That is a quieter wrong answer than the TypeError being fixed here.
    """

    def search(self, query, **kwargs):
        self.query = query
        self.sent = dict(kwargs)
        return _response()


@pytest.fixture
def build_retriever(monkeypatch):
    """Install a fake `exa_py` module, then build an ExaSearch against it.

    The fake module needs a real __spec__: retrievers.utils.check_pkg gates on
    importlib.util.find_spec, which raises ValueError on a spec-less sys.modules entry.
    """
    def _build(client_cls, **kwargs):
        module = types.ModuleType("exa_py")
        module.__spec__ = ModuleSpec("exa_py", None)
        module.Exa = client_cls
        monkeypatch.setitem(sys.modules, "exa_py", module)
        monkeypatch.setenv("EXA_API_KEY", "not-a-real-key")
        return ExaSearch(QUERY, **kwargs)

    return _build


def test_the_retriever_does_not_send_an_argument_the_client_dropped(build_retriever):
    """The measured failure: one retired argument kills every exa call."""
    retriever = build_retriever(_Exa220, query_domains=DOMAINS)

    retriever.search(max_results=6, search_type="neural")

    assert "use_autoprompt" not in retriever.client.sent, (
        "the retriever sent use_autoprompt to a client that does not declare it; the "
        "installed exa-py 2.20.0 answers TypeError: Exa.search() got an unexpected "
        "keyword argument 'use_autoprompt', and after two such failures SmartRetriever "
        "retires exa from routing for the whole process"
    )


def test_the_retriever_still_sends_what_the_client_does_declare(build_retriever):
    """Dropping the dead argument must not cost the three arguments that carry meaning."""
    retriever = build_retriever(_Exa220, query_domains=DOMAINS)

    retriever.search(max_results=6, search_type="neural")

    sent = retriever.client.sent
    assert retriever.client.query == QUERY, (
        f"the client was queried for {retriever.client.query!r}, not the retriever's "
        f"own query {QUERY!r}"
    )
    assert sent.get("type") == "neural", (
        f"the search mode reached the client as {sent.get('type')!r}; the code_technical "
        "route asks exa for a neural search and a keyword search answers differently"
    )
    assert sent.get("num_results") == 6, (
        f"the result limit reached the client as {sent.get('num_results')!r} instead of "
        "6, so the route's per-retriever budget is not what exa is asked for"
    )
    assert sent.get("include_domains") == DOMAINS, (
        f"domain filtering reached the client as {sent.get('include_domains')!r} instead "
        f"of {DOMAINS}, so a domain-scoped sub-query searches the open web"
    )


def test_a_client_that_drops_one_more_argument_does_not_retire_the_retriever(build_retriever):
    """Pins the filter as dynamic: a hard-coded 2.20 list passes the previous test too."""
    retriever = build_retriever(_ExaNarrower, query_domains=DOMAINS)

    results = retriever.search(max_results=6, search_type="neural")

    sent = retriever.client.sent
    assert "type" not in sent, (
        "the retriever sent type= to a client whose search() does not declare it, which "
        "is the same TypeError that retired exa on 2.20.0 - the argument list is pinned "
        "to one version instead of read off the client at call time"
    )
    assert sent.get("num_results") == 6 and sent.get("include_domains") == DOMAINS, (
        f"dropping the unsupported argument also lost the supported ones: {sent}"
    )
    assert len(results) == 2, (
        f"the narrower client returned 2 results but the retriever surfaced {len(results)}"
    )


def test_a_client_that_declares_kwargs_is_still_given_the_whole_intent(build_retriever):
    """The filter must not mistake "accepts anything" for "accepts nothing"."""
    retriever = build_retriever(_ExaVarKeyword, query_domains=DOMAINS)

    retriever.search(max_results=6, search_type="neural")

    assert retriever.client.sent == {
        "type": "neural", "num_results": 6, "include_domains": DOMAINS,
    }, (
        f"a **kwargs client was sent {retriever.client.sent}; it declares no parameter "
        "names, so filtering by name drops the search mode, the result limit and the "
        "domain filter, and exa answers with unfiltered defaults and no error at all"
    )


def test_a_result_carrying_no_contents_still_has_a_string_body(build_retriever):
    """exa-py 2.20 search() leaves Result.text None unless contents were requested.

    Every other retriever in the fork returns a string body, and SmartRetriever's
    de-duplication does len(r.get("body", "")) - the key EXISTS, so the "" default never
    fires and a None body raises TypeError the moment an exa URL collides with the
    serper or github result in the same code_technical bundle. That kills the whole
    query's search, not just exa's share of it.
    """
    retriever = build_retriever(_Exa220, query_domains=DOMAINS)

    results = retriever.search(max_results=6, search_type="neural")

    bad = [r for r in results if not isinstance(r["body"], str)]
    assert not bad, (
        f"{len(bad)} of {len(results)} exa results carry a non-string body "
        f"({[r['body'] for r in bad]}); SmartRetriever._deduplicate_results calls len() "
        "on it and raises TypeError as soon as two retrievers in the bundle return the "
        "same URL"
    )
