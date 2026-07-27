"""s1 — a swallowed retriever failure must leave an ERROR record.

Measured in the round-2 live probe: retriever_errors read 0 for a deployment whose
tavily answers 432 to every call. Two holes produced that reading, and both are the
same defect — a failure that only exists as a WARNING (or a stdout print) is
invisible to the probe, which counts ERROR-level log records:

  (a) SmartRetriever swallows each per-retriever exception into an empty list; when
      a sibling retriever in the same parallel bundle returns anything, the routed
      retriever's total outage is indistinguishable from a healthy run.
  (b) every retriever that catches its own exception printed it to stdout instead of
      logging it — duckduckgo, the keyless head of the fallback order, included.

Deterministic: no network, no LLM. requests.post is mocked and ddgs is replaced by
an in-memory fake module.
"""
import importlib.machinery
import logging
import sys
import types
from types import SimpleNamespace

import requests


class _Resp432:
    status_code = 432

    def raise_for_status(self):
        raise requests.HTTPError("432 Client Error: rate limited")

    def json(self):
        return {}


def test_tavily_432_emits_error_record_even_when_a_fallback_covers_for_it(monkeypatch, caplog):
    import gpt_researcher.actions.retriever as retriever_actions
    from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp432())

    fallback_results = [{"href": "https://fallback.example.com/a", "body": "fallback body A"}]

    class _FakeFallbackRetriever:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return fallback_results

    def fake_get_retriever(name):
        return TavilySearch if name == "tavily" else _FakeFallbackRetriever

    monkeypatch.setattr(retriever_actions, "get_retriever", fake_get_retriever)

    cfg = SimpleNamespace(
        smart_retriever_force_category="general_web",
        smart_retriever_config={"general_web": [("tavily", 7, {})]},
    )
    with caplog.at_level(logging.ERROR):
        results = SmartRetriever("some query", cfg=cfg).search(max_results=10)

    assert {r.get("href") for r in results} == {"https://fallback.example.com/a"}, (
        "precondition: the fallback still has to cover for the 432"
    )
    errors = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert any("tavily" in msg for msg in errors), (
        "a retriever that fails on every call must leave an ERROR record naming it — "
        "at WARNING the live probe (an ERROR-level handler) reads 0 retriever errors "
        f"and a total tavily outage passes as healthy. records: {errors}"
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
