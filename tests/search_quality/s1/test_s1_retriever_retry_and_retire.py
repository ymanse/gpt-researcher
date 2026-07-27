"""s1 tests — defect 1: a 432 must be retried, then routed around, and said out loud.

Measured on this deployment: TavilySearch raises HTTPError 432 for every query, yet
the s1 live probe (which counts ERROR log records) read retriever_errors=0 the whole
time, because a sibling retriever in the same parallel bundle returned something.

The contract these tests pin:
  - one failure is retried, and a retry that succeeds is NOT an error (a transient
    blip must not read as a broken retriever);
  - a retriever that fails twice is reported ONCE and retired from routing for the
    process, so the next sub-query routes to an alternate instead of re-paying for
    the same failure. That report is a WARNING while a sibling covers the query:
    spec/search-quality.md s1 counts UNRECOVERED failures only, and the ERROR is
    emitted by search() when the query ends up with nothing (see
    test_s1_retriever_error_signal.py).

Deterministic: no network, retrievers are fakes.
"""
import logging
from types import SimpleNamespace

import gpt_researcher.actions.retriever as retriever_actions
from gpt_researcher.retrievers.smart import smart_retriever as smart_mod
from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever

GOOD = [{"href": "https://alt.example/a", "body": "alternate body"}]


def _cfg(route):
    return SimpleNamespace(smart_retriever_force_category="general_web",
                           smart_retriever_config={"general_web": route})


def _fake_registry(monkeypatch, classes):
    monkeypatch.setattr(retriever_actions, "get_retriever", lambda name: classes[name])


def _flaky(fail_times, results=GOOD):
    """A retriever class whose search() raises for the first `fail_times` calls."""
    state = {"calls": 0}

    class _Flaky:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise RuntimeError("432 Client Error: rate limited")
            return list(results)

    return _Flaky, state


def test_a_failure_the_retry_clears_is_not_an_error(monkeypatch, caplog):
    flaky, state = _flaky(fail_times=1)
    _fake_registry(monkeypatch, {"tavily": flaky})

    with caplog.at_level(logging.DEBUG):
        results = SmartRetriever("q", cfg=_cfg([("tavily", 5, {})])).search(max_results=5)

    assert results == GOOD, "the retry's results must be used, not discarded"
    assert state["calls"] == 2, "the failed call must be retried exactly once"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "a transient failure the retry cleared is not a broken retriever — reporting "
        "it at ERROR would fail the live gate on every blip"
    )
    assert "tavily" not in smart_mod._DEAD_RETRIEVERS


def test_a_retriever_that_fails_twice_is_reported_once_and_retired(monkeypatch, caplog):
    dead, dead_state = _flaky(fail_times=99)
    alt, alt_state = _flaky(fail_times=0)
    _fake_registry(monkeypatch, {"tavily": dead, "duckduckgo": alt})
    route = [("tavily", 5, {}), ("duckduckgo", 5, {})]

    def _retirement_records():
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "retired" in r.getMessage()]

    with caplog.at_level(logging.DEBUG):
        first = SmartRetriever("q", cfg=_cfg(route)).search(max_results=5)
        retirements_after_first = _retirement_records()
        second = SmartRetriever("q2", cfg=_cfg(route)).search(max_results=5)

    assert first == GOOD and second == GOOD, (
        "the alternate must cover the query both times — a dead retriever is routed "
        "around, not propagated as a total loss"
    )
    assert len(retirements_after_first) == 1, (
        "a retriever that cannot serve must be reported EXACTLY once: silence hides "
        "it entirely, and one record per sub-query buries the log. got "
        f"{retirements_after_first}"
    )
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "the sibling covered both queries, so nothing here is an UNRECOVERED failure "
        "— an ERROR would fail the live gate on a search that succeeded. got "
        f"{[r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]}"
    )
    assert "tavily" in smart_mod._DEAD_RETRIEVERS
    assert dead_state["calls"] == 2, (
        "after being retired the retriever must not be called again — the second "
        f"query re-paid for the same failure ({dead_state['calls']} calls)"
    )
    assert alt_state["calls"] == 2, "the alternate still serves every query"
    assert len(_retirement_records()) == 1, (
        "the retirement is reported once for the whole process, not once per query"
    )


def test_route_kwargs_survive_being_run_twice(monkeypatch):
    """The route's extra_kwargs dict lives in ROUTING_TABLE; consuming it in place
    would strip query_domains from every later query in the process."""
    seen = []

    class _Recorder:
        def __init__(self, query, query_domains=None, **kwargs):
            seen.append(query_domains)

        def search(self, max_results=5, **kwargs):
            return list(GOOD)

    _fake_registry(monkeypatch, {"serper": _Recorder})
    route = [("serper", 4, {"query_domains": ["github.com"]})]
    cfg = _cfg(route)

    SmartRetriever("q", cfg=cfg).search(max_results=5)
    SmartRetriever("q2", cfg=cfg).search(max_results=5)

    assert seen == [["github.com"], ["github.com"]], (
        f"the route's query_domains must not be consumed by the first run — got {seen}"
    )
