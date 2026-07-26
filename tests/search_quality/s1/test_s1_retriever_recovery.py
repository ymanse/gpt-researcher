"""RED tests for s1 — retriever recovery (spec/search-quality.md, observed defects 1-2).

Covers the three OBSERVED failures from container logs:
  (a) wikipedia language code parsed as "wt" (ddgs splits region "wt-wt" into
      country/lang and requests https://wt.wikipedia.org -> DNS fail): the code
      we ship must normalize to a valid wikipedia language code.
  (b) tavily HTTP 432 is swallowed into a silent empty result set: it must
      raise a detectable error and trigger fallback routing to another retriever.
  (c) a scrape pass that ends with 0 URLs / 0 pages passes silently: it must
      emit a retriever/scraper ERROR signal (the s1 live probe counts ERROR
      records — a healthy run has none, an empty-handed pass must have one).

Deterministic, no network: retriever HTTP (requests.post) and the scrape pass
are mocked; ddgs is replaced by an in-memory fake module.
"""
import importlib.machinery
import logging
import re
import sys
import types
from types import SimpleNamespace

import pytest
import requests


# ---------------------------------------------------------------------------
# (a) wikipedia language-code normalization
# ---------------------------------------------------------------------------

def test_wikipedia_lang_wt_normalized_to_valid_code():
    # "wt" is the observed poison value ("wt-wt" region -> wt.wikipedia.org DNS fail)
    from gpt_researcher.retrievers.utils import normalize_wikipedia_lang

    assert normalize_wikipedia_lang("wt") == "en"
    assert normalize_wikipedia_lang("wt-wt") == "en"


def test_wikipedia_lang_valid_codes_pass_through():
    from gpt_researcher.retrievers.utils import normalize_wikipedia_lang

    assert normalize_wikipedia_lang("en") == "en"
    assert normalize_wikipedia_lang("ko") == "ko"
    # region form country-lang: the language segment is what wikipedia needs
    assert normalize_wikipedia_lang("us-en") == "en"


def test_wikipedia_lang_garbage_falls_back_to_en():
    from gpt_researcher.retrievers.utils import normalize_wikipedia_lang

    assert normalize_wikipedia_lang("") == "en"
    assert normalize_wikipedia_lang(None) == "en"


def test_duckduckgo_region_language_segment_is_valid_for_wikipedia(monkeypatch):
    """ddgs fans one region out to every engine; its wikipedia engine does
    `_country, lang = region.lower().split("-")` and builds
    https://{lang}.wikipedia.org. The region our retriever passes must therefore
    carry a real language code in the lang segment — never "wt"."""
    captured = {}
    fallback = [{"href": "https://example.com/ddg", "body": "ddg body"}]

    fake_mod = types.ModuleType("ddgs")
    fake_mod.__spec__ = importlib.machinery.ModuleSpec("ddgs", loader=None)

    class _FakeDDGS:
        def text(self, query, region=None, max_results=None, **kwargs):
            captured["region"] = region
            return fallback

    fake_mod.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)

    from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo

    results = Duckduckgo("test query").search(max_results=3)

    region = captured["region"]
    parts = region.lower().split("-")
    assert len(parts) == 2, f"region {region!r} must be country-lang form"
    lang = parts[1]
    assert re.fullmatch(r"[a-z]{2,3}", lang), f"lang segment {lang!r} not a language code"
    assert lang != "wt", "observed defect: lang 'wt' -> wt.wikipedia.org DNS fail"
    assert results, "normalized region must not break the search flow"


# ---------------------------------------------------------------------------
# (b) tavily 432 -> fallback routing, not silent total loss
# ---------------------------------------------------------------------------

class _Resp432:
    status_code = 432

    def raise_for_status(self):
        raise requests.HTTPError("432 Client Error: rate limited")

    def json(self):
        return {}


def test_tavily_432_raises_instead_of_silent_empty(monkeypatch):
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp432())

    with pytest.raises(Exception):
        # today: the 432 is caught, printed, and silently turned into []
        TavilySearch("some query").search(max_results=5)


def test_tavily_432_triggers_fallback_routing_to_another_retriever(monkeypatch):
    import gpt_researcher.actions.retriever as retriever_actions
    from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp432())

    fallback_results = [
        {"href": "https://fallback.example.com/a", "body": "fallback body A"},
        {"href": "https://fallback.example.com/b", "body": "fallback body B"},
    ]

    class _FakeFallbackRetriever:
        def __init__(self, query, query_domains=None, **kwargs):
            self.query = query

        def search(self, max_results=5, **kwargs):
            return fallback_results

    def fake_get_retriever(name):
        if name == "tavily":
            return TavilySearch  # real class -> real 432 handling path
        return _FakeFallbackRetriever  # any fallback choice yields results

    monkeypatch.setattr(retriever_actions, "get_retriever", fake_get_retriever)

    cfg = SimpleNamespace(
        smart_retriever_force_category="general_web",
        # tavily-only route mirrors the observed outage: every routed retriever 432s
        smart_retriever_config={"general_web": [("tavily", 7, {})]},
    )
    results = SmartRetriever("some query", cfg=cfg).search(max_results=10)

    hrefs = {r.get("href") for r in results}
    assert "https://fallback.example.com/a" in hrefs, (
        "tavily 432 must trigger fallback routing to another retriever "
        "instead of a silent total loss (empty result set)"
    )


# ---------------------------------------------------------------------------
# (c) scrape pass returning 0 URLs/pages must raise an error signal
# ---------------------------------------------------------------------------

def _make_researcher():
    cfg = SimpleNamespace(
        max_scraper_workers=2,
        scraper_rate_limit_delay=0.0,
        scraper="bs",
        user_agent="pytest-agent",
    )
    r = SimpleNamespace(
        cfg=cfg,
        verbose=False,
        websocket=None,
        vector_store=None,
        research_sources=[],
        research_images=[],
    )
    r.add_research_sources = lambda sources: r.research_sources.extend(sources)
    r.add_research_images = lambda images: r.research_images.extend(images)
    r.get_research_images = lambda: r.research_images
    return r


async def test_scrape_pass_with_zero_urls_emits_error_signal(caplog):
    from gpt_researcher.skills.browser import BrowserManager

    manager = BrowserManager(_make_researcher())
    with caplog.at_level(logging.ERROR):
        scraped = await manager.browse_urls([])

    assert scraped == []
    error_records = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert error_records, (
        "a scrape pass given 0 URLs must emit a retriever/scraper ERROR signal, "
        "not pass silently"
    )


async def test_scrape_pass_yielding_zero_pages_emits_error_signal(monkeypatch, caplog):
    import gpt_researcher.skills.browser as browser_mod

    async def fake_scrape_urls(urls, cfg, worker_pool):
        return [], []  # scraper came back empty-handed for every URL

    monkeypatch.setattr(browser_mod, "scrape_urls", fake_scrape_urls)

    manager = browser_mod.BrowserManager(_make_researcher())
    with caplog.at_level(logging.ERROR):
        scraped = await manager.browse_urls(
            ["https://example.com/x", "https://example.com/y"]
        )

    assert scraped == []
    error_records = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert error_records, (
        "a scrape pass that yields 0 pages must emit a retriever/scraper ERROR "
        "signal, not pass silently"
    )
