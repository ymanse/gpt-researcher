"""Stage 3 RED tests — citation-verification pass (CitationAgent).

Target: a lightweight CitationAgent that re-fetches each core learning's
citation URL via firecrawl /v2/scrape with maxAge=0 (bypass cache), then
matches the cited quote against the fetched markdown (string containment
with whitespace/case normalization). Unmatched or unfetchable claims are
flagged unverified. The result reports total_claims / grounded / unverified
consistently (this shape feeds the stage 3 smoke evidence JSON).

Input contract: the citations dict produced by DeepResearchSkill.
process_research_results — {quote/learning text: citation url}.

Deterministic, no network: firecrawl scrape HTTP is mocked via requests.post.
Imports happen INSIDE tests so a missing implementation is a test FAILURE,
not a collection error (RED gate requires errors=0, failed>=1).
"""
from unittest import mock

import pytest

# Per-URL mocked scrape sources. Quote A lives in a.example.com, b.example.com
# does NOT contain quote B, c.example.com fails to scrape.
SCRAPED_MARKDOWN = {
    "https://a.example.com/grid": (
        "# Grid storage\n\n"
        "Utilities report that the grid stores excess solar energy in batteries "
        "during peak production hours.\n\n## Outlook\n\nMore capacity is planned.\n"
    ),
    "https://b.example.com/solar": (
        "# Solar panels\n\nThis page talks about panel efficiency only, "
        "nothing about storage claims.\n"
    ),
}


def _fake_post(url=None, *args, **kwargs):
    """requests.post double for firecrawl /v2/scrape, keyed by payload url."""
    payload = kwargs.get("json") or {}
    target = payload.get("url", "")
    resp = mock.MagicMock()
    resp.raise_for_status.return_value = None
    if target in SCRAPED_MARKDOWN:
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"markdown": SCRAPED_MARKDOWN[target]},
        }
    else:
        resp.status_code = 200
        resp.json.return_value = {"success": False, "data": {}}
    return resp


@pytest.fixture(autouse=True)
def firecrawl_key(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")


def _import_agent():
    try:
        from gpt_researcher.skills.citation_verification import CitationAgent
    except ImportError as e:
        pytest.fail(f"CitationAgent not implemented yet: {e}")
    return CitationAgent


def _verify(citations):
    CitationAgent = _import_agent()
    with mock.patch("requests.post", side_effect=_fake_post) as post:
        result = CitationAgent().verify(citations)
    return result, post


# ---------------------------------------------------------------------------
# (a) quote present in mocked source -> verified/grounded
# ---------------------------------------------------------------------------

class TestQuoteGrounded:
    def test_exact_quote_verified(self):
        result, _ = _verify(
            {"the grid stores excess solar energy in batteries": "https://a.example.com/grid"}
        )
        assert result["total_claims"] == 1
        assert result["grounded"] == 1
        assert result["unverified"] == 0
        (claim,) = result["claims"]
        assert claim["url"] == "https://a.example.com/grid"
        assert claim["verified"] is True

    def test_normalized_match_case_and_whitespace(self):
        # containment must survive case + whitespace differences vs the source
        result, _ = _verify(
            {"The  Grid   stores excess\nsolar energy in batteries": "https://a.example.com/grid"}
        )
        assert result["grounded"] == 1
        assert result["claims"][0]["verified"] is True

    def test_scrape_uses_v2_scrape_with_max_age_zero(self):
        _, post = _verify(
            {"the grid stores excess solar energy in batteries": "https://a.example.com/grid"}
        )
        assert post.called
        args, kwargs = post.call_args
        url = kwargs.get("url") or (args[0] if args else "")
        assert "/v2/scrape" in url
        payload = kwargs.get("json") or {}
        assert payload.get("url") == "https://a.example.com/grid"
        assert payload.get("maxAge") == 0, "re-fetch must bypass firecrawl cache (maxAge=0)"


# ---------------------------------------------------------------------------
# (b) quote absent -> flagged unverified
# ---------------------------------------------------------------------------

class TestQuoteUnverified:
    def test_absent_quote_flagged(self):
        result, _ = _verify(
            {"batteries now cost under ten dollars per kWh": "https://b.example.com/solar"}
        )
        assert result["total_claims"] == 1
        assert result["grounded"] == 0
        assert result["unverified"] == 1
        (claim,) = result["claims"]
        assert claim["verified"] is False

    def test_unfetchable_source_flagged(self):
        result, _ = _verify(
            {"any claim at all": "https://gone.example.com/404"}
        )
        assert result["grounded"] == 0
        assert result["unverified"] == 1
        assert result["claims"][0]["verified"] is False


# ---------------------------------------------------------------------------
# (c) counts reported: total_claims / grounded / unverified consistent
# ---------------------------------------------------------------------------

class TestCountsConsistent:
    def test_mixed_claims_counts(self):
        citations = {
            "the grid stores excess solar energy in batteries": "https://a.example.com/grid",
            "More capacity is planned": "https://a.example.com/grid",
            "batteries now cost under ten dollars per kWh": "https://b.example.com/solar",
            "any claim at all": "https://gone.example.com/404",
        }
        result, _ = _verify(citations)
        assert result["total_claims"] == 4
        assert result["grounded"] == 2
        assert result["unverified"] == 2
        assert result["total_claims"] == result["grounded"] + result["unverified"]
        assert len(result["claims"]) == 4
        assert sum(c["verified"] for c in result["claims"]) == result["grounded"]

    def test_empty_citations(self):
        result, post = _verify({})
        assert result == {"total_claims": 0, "grounded": 0, "unverified": 0, "claims": []}
        assert not post.called
