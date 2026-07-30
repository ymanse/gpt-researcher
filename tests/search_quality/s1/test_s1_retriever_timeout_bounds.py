"""s1 tests — the parallel-bundle timeout must actually bound the wait.

Measured on this deployment (gptr-mcp-server, 2026-07-29 deep_research run):

    06:26:19  WARNING Retrievers timed out: ['arxiv']. Returning partial results.
    06:31:29  INFO    Initial search results obtained: 8 results

The timeout said it gave up at 06:26:19, but the call did not return for another
5m10s. `_execute_retrievers` ran its pool inside `with ThreadPoolExecutor(...)`,
and leaving that block calls shutdown(wait=True) — which joins the very straggler
the timeout just declared abandoned. The whole 37-minute run spent ~6.5 minutes
waiting on a retriever whose results were already written off.

The contract these tests pin:
  - the bundle returns within its timeout, keeping whatever siblings produced;
  - a retriever that times out twice is retired, so the next sub-query routes to
    an alternate instead of re-paying the timeout on every one of ~39 queries.

Deterministic: no network, retrievers are fakes, timeout patched down to ~1s.
"""
import time
from types import SimpleNamespace

import gpt_researcher.actions.retriever as retriever_actions
from gpt_researcher.retrievers.smart import smart_retriever as smart_mod
from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever

FAST = [{"href": "https://fast.example/a", "body": "fast body"}]


def _cfg(route):
    return SimpleNamespace(smart_retriever_force_category="general_web",
                           smart_retriever_config={"general_web": route})


def _fake_registry(monkeypatch, classes):
    monkeypatch.setattr(retriever_actions, "get_retriever", lambda name: classes[name])


def _sleeper(seconds):
    """A retriever whose search() blocks far past the bundle timeout."""

    class _Sleeper:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            time.sleep(seconds)
            return []

    return _Sleeper


def _fast():
    class _Fast:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return list(FAST)

    return _Fast


def test_a_hung_retriever_does_not_hold_the_bundle_past_the_timeout(monkeypatch):
    monkeypatch.setattr(smart_mod, "_RETRIEVER_TIMEOUT_S", 1.0)
    _fake_registry(monkeypatch, {"duckduckgo": _fast(), "arxiv": _sleeper(20)})
    retriever = SmartRetriever("q", cfg=_cfg([("duckduckgo", 5, {}), ("arxiv", 5, {})]))

    started = time.monotonic()
    results = retriever._execute_retrievers([("duckduckgo", 5, {}), ("arxiv", 5, {})])
    elapsed = time.monotonic() - started

    # The sibling's results survive; the straggler is abandoned, not joined.
    assert results == FAST
    assert elapsed < 5, f"bundle blocked {elapsed:.1f}s on an abandoned retriever"


def test_a_retriever_that_times_out_twice_is_retired(monkeypatch):
    monkeypatch.setattr(smart_mod, "_RETRIEVER_TIMEOUT_S", 1.0)
    _fake_registry(monkeypatch, {"duckduckgo": _fast(), "arxiv": _sleeper(20)})
    route = [("duckduckgo", 5, {}), ("arxiv", 5, {})]
    retriever = SmartRetriever("q", cfg=_cfg(route))

    retriever._execute_retrievers(route)
    assert "arxiv" not in smart_mod._DEAD_RETRIEVERS, "one timeout may be a blip"

    retriever._execute_retrievers(route)
    assert "arxiv" in smart_mod._DEAD_RETRIEVERS, (
        "a second timeout is a pattern: retire it so the remaining sub-queries "
        "do not each re-pay the timeout"
    )
