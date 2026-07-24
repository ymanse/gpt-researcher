"""Stage 1 RED tests — Firecrawl /v2/search retriever + SmartRetriever routing.

Deterministic, no network: the Firecrawl HTTP API is mocked via requests.post.
Adapter imports happen INSIDE tests so a missing implementation is a test
FAILURE, not a collection error (RED gate requires errors=0, failed>=1).
"""
from unittest import mock

import pytest

# Mocked Firecrawl /v2/search response (scrapeOptions.formats=["markdown"]
# makes each web result carry full-page markdown).
FIRECRAWL_V2_SEARCH_RESPONSE = {
    "success": True,
    "data": {
        "web": [
            {
                "url": "https://example.com/solid-state-batteries",
                "title": "Solid-state battery manufacturing in 2026",
                "description": "Short snippet only.",
                "markdown": (
                    "# Solid-state battery manufacturing\n\n"
                    "Full clean markdown body extracted from the page, "
                    "substantially longer than a search snippet.\n\n"
                    "## Key developments\n\n- pilot lines scaled\n- yields improved\n"
                ),
            },
            {
                "url": "https://news.example.org/battery-plants",
                "title": "New battery plants announced",
                "description": "Another snippet.",
                "markdown": "## Battery plants\n\nMarkdown content for the second result.\n",
            },
        ]
    },
}


def _make_response(payload):
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def firecrawl_key(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")


@pytest.fixture
def no_firecrawl_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# (a) FirecrawlSearch adapter normalizes /v2/search into [{href,title,body}]
# ---------------------------------------------------------------------------

class TestFirecrawlAdapter:
    def _import_adapter(self):
        try:
            from gpt_researcher.retrievers.firecrawl import FirecrawlSearch
        except ImportError as e:
            pytest.fail(f"FirecrawlSearch adapter not implemented yet: {e}")
        return FirecrawlSearch

    def test_search_normalizes_v2_response(self, firecrawl_key):
        FirecrawlSearch = self._import_adapter()
        with mock.patch("requests.post", return_value=_make_response(FIRECRAWL_V2_SEARCH_RESPONSE)):
            results = FirecrawlSearch(query="solid-state batteries").search(max_results=5)

        assert isinstance(results, list) and len(results) == 2
        for r in results:
            assert set(r) >= {"href", "title", "body"}
            assert r["href"].startswith("https://")
            assert r["title"]
            # body must be the clean markdown, not the short snippet
            assert r["body"].strip()
        assert results[0]["href"] == "https://example.com/solid-state-batteries"
        assert results[0]["title"] == "Solid-state battery manufacturing in 2026"
        assert "# Solid-state battery manufacturing" in results[0]["body"]
        assert results[0]["body"] != "Short snippet only."

    def test_search_requests_markdown_scrape_format(self, firecrawl_key):
        FirecrawlSearch = self._import_adapter()
        with mock.patch("requests.post", return_value=_make_response(FIRECRAWL_V2_SEARCH_RESPONSE)) as post:
            FirecrawlSearch(query="solid-state batteries").search(max_results=5)

        assert post.called
        _, kwargs = post.call_args
        payload = kwargs.get("json") or {}
        assert payload.get("scrapeOptions", {}).get("formats") == ["markdown"]
        url = kwargs.get("url") or (post.call_args[0][0] if post.call_args[0] else "")
        assert "/v2/search" in url

    def test_search_skips_results_without_markdown_body(self, firecrawl_key):
        FirecrawlSearch = self._import_adapter()
        payload = {
            "success": True,
            "data": {
                "web": [
                    {"url": "https://a.example.com", "title": "no body", "markdown": ""},
                    {"url": "https://b.example.com", "title": "has body", "markdown": "Real markdown.\n"},
                ]
            },
        }
        with mock.patch("requests.post", return_value=_make_response(payload)):
            results = FirecrawlSearch(query="q").search(max_results=5)

        assert all(r["body"].strip() for r in results)
        assert [r["href"] for r in results] == ["https://b.example.com"]


# ---------------------------------------------------------------------------
# (b) SmartRetriever routing includes firecrawl + API-key mapping
# ---------------------------------------------------------------------------

class TestSmartRetrieverRouting:
    @pytest.mark.parametrize("category", ["general_web", "comprehensive", "news_current"])
    def test_routing_table_includes_firecrawl(self, category):
        from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE

        names = [entry[0] for entry in ROUTING_TABLE[category]]
        assert "firecrawl" in names, f"'{category}' routing must include firecrawl"

    def test_api_key_mapping(self):
        from gpt_researcher.retrievers.smart.smart_retriever import _RETRIEVER_API_KEYS

        assert _RETRIEVER_API_KEYS.get("firecrawl") == "FIRECRAWL_API_KEY"

    def test_firecrawl_registered_in_retriever_factory(self):
        from gpt_researcher.actions.retriever import get_retriever

        cls = get_retriever("firecrawl")
        assert cls is not None, "get_retriever('firecrawl') must resolve the adapter"
        assert cls.__name__ == "FirecrawlSearch"


# ---------------------------------------------------------------------------
# (c) Without FIRECRAWL_API_KEY the firecrawl retriever is excluded
# ---------------------------------------------------------------------------

class TestFirecrawlAvailability:
    def _smart(self):
        from gpt_researcher.retrievers.smart.smart_retriever import SmartRetriever

        return SmartRetriever(query="anything")

    def test_availability_requires_key(self, no_firecrawl_key, monkeypatch):
        smart = self._smart()
        assert smart._check_retriever_availability("firecrawl") is False

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        assert smart._check_retriever_availability("firecrawl") is True

    def test_routing_excludes_firecrawl_without_key(self, no_firecrawl_key, monkeypatch):
        smart = self._smart()
        assert "firecrawl" not in [e[0] for e in smart._route_to_retrievers("general_web")]

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        assert "firecrawl" in [e[0] for e in smart._route_to_retrievers("general_web")], (
            "with the key set, existing availability logic must route firecrawl for general_web"
        )
