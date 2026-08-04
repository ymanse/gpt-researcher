"""s1 — a retry only makes sense if the failure could plausibly clear on a second try.

Observed live (harness-search): GithubSearch logged
`HTTP Error 422: Unprocessable Entity. Failed fetching GitHub sources.` six times —
once per sub-query, because a 422 means the request itself is malformed and every
retry resends the identical request into the identical rejection. Firecrawl's
`502 Server Error: Bad Gateway` is the opposite case: a gateway hiccup that a second
call can clear.

The contract these tests pin:
  - retrievers.utils.is_retryable_error classifies 400/401/403/404/422 as NOT
    retryable and 429/502/503/504 (plus any status-less/network failure) as
    retryable — spec/search-quality.md task 3 (2026-08-04);
  - SmartRetriever spends the one free retry ONLY on a retryable failure. A
    non-retryable one retires the retriever on the FIRST failure — no wasted
    second call — while a retryable one still gets the existing retry-then-retire
    treatment (test_s1_retriever_retry_and_retire.py, unchanged).

Deterministic: no network, retrievers are fakes.
"""
from types import SimpleNamespace

import pytest
import requests

import gpt_researcher.actions.retriever as retriever_actions
from gpt_researcher.retrievers.smart import smart_retriever as smart_mod
from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever
from gpt_researcher.retrievers.utils import is_retryable_error

GOOD = [{"href": "https://alt.example/a", "body": "alternate body"}]


# ---------------------------------------------------------------------------
# is_retryable_error: the classification itself
# ---------------------------------------------------------------------------

def _http_error(status_code):
    return requests.HTTPError(f"{status_code} error", response=SimpleNamespace(status_code=status_code))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert is_retryable_error(_http_error(status)) is False


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_gateway_and_rate_limit_errors_are_retryable(status):
    assert is_retryable_error(_http_error(status)) is True


def test_status_less_exception_is_retryable():
    # No response was ever received (connection refused, DNS failure, read
    # timeout) — transient by definition, and also preserves today's behavior
    # for a code with no evidence either way (e.g. Tavily's non-standard 432).
    assert is_retryable_error(RuntimeError("connection refused")) is True
    assert is_retryable_error(_http_error(432)) is True


# ---------------------------------------------------------------------------
# SmartRetriever wiring: the outcome that actually matters
# ---------------------------------------------------------------------------

def _cfg(route):
    return SimpleNamespace(smart_retriever_force_category="general_web",
                            smart_retriever_config={"general_web": route})


def _fake_registry(monkeypatch, classes):
    monkeypatch.setattr(retriever_actions, "get_retriever", lambda name: classes[name])


def _raiser(exc):
    """A retriever class whose search() always raises `exc`."""
    state = {"calls": 0}

    class _Raiser:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            state["calls"] += 1
            raise exc

    return _Raiser, state


def _ok():
    class _Ok:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return list(GOOD)

    return _Ok


def test_a_422_is_not_retried_and_retires_on_the_first_failure(monkeypatch, caplog):
    github, state = _raiser(_http_error(422))
    _fake_registry(monkeypatch, {"github": github, "duckduckgo": _ok()})
    route = [("github", 5, {}), ("duckduckgo", 5, {})]

    results = SmartRetriever("q", cfg=_cfg(route)).search(max_results=5)

    assert results == GOOD, "the sibling must still cover the query"
    assert state["calls"] == 1, (
        f"a 422 must fail fast — no retry, since the identical request gets the "
        f"identical rejection. got {state['calls']} calls"
    )
    assert "github" in smart_mod._DEAD_RETRIEVERS, (
        "a non-retryable failure must retire the retriever on the FIRST failure, "
        "not the second"
    )
    assert any("without a wasted retry" in r.getMessage() for r in caplog.records), (
        "the fail-fast path must say so, not look identical to the retry-then-retire path"
    )


def test_a_502_still_gets_the_one_free_retry_before_retiring(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")  # availability-gated, see _RETRIEVER_API_KEYS
    firecrawl, state = _raiser(_http_error(502))
    _fake_registry(monkeypatch, {"firecrawl": firecrawl, "duckduckgo": _ok()})
    route = [("firecrawl", 5, {}), ("duckduckgo", 5, {})]

    results = SmartRetriever("q", cfg=_cfg(route)).search(max_results=5)

    assert results == GOOD
    assert state["calls"] == 2, (
        f"a 502 is a gateway hiccup a retry can clear — it must get the existing "
        f"retry-then-retire treatment, not fail-fast. got {state['calls']} calls"
    )
    assert "firecrawl" in smart_mod._DEAD_RETRIEVERS


def test_github_search_raises_instead_of_swallowing_http_errors(monkeypatch):
    """GithubSearch must surface HTTP failures (like tavily already does) so
    SmartRetriever can classify and route around them — swallowed, they read
    as an empty result and never get retired."""
    import urllib.error

    from gpt_researcher.retrievers.github.github import GithubSearch

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable Entity", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        GithubSearch("some query").search(max_results=5)


def test_firecrawl_search_raises_instead_of_swallowing_http_errors(monkeypatch):
    """FirecrawlSearch must surface HTTP failures (like tavily already does)."""
    from gpt_researcher.retrievers.firecrawl.firecrawl import FirecrawlSearch

    class _Resp502:
        status_code = 502

        def raise_for_status(self):
            raise requests.HTTPError("502 Server Error: Bad Gateway", response=self)

        def json(self):
            return {}

    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp502())

    with pytest.raises(requests.HTTPError):
        FirecrawlSearch("some query").search(max_results=5)
