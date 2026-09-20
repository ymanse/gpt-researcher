"""The Crawl4AI scraper reads the server's envelope, and never takes the batch down.

Crawl4AI replaces the SCRAPE half of Firecrawl: Firecrawl's free tier hit 0 on
2026-09-20 with ten days left in the period, and its key is shared with the Claude Code
firecrawl MCP, so research runs and interactive sessions drew on the same 1,000/month.
The local container has no quota. What it does NOT replace is Firecrawl's search
retriever -- Crawl4AI takes a URL and returns that page, and has no search endpoint.

Two things are pinned here, both of which cost a real debugging session to learn:

1. The envelope. 0.9.3 answers `{"success": ..., "results": [ {...} ]}` and inside the
   result `markdown` is an OBJECT (`raw_markdown`, `fit_markdown`, ...), not a string.
   Reading `result["markdown"]` as text yields a dict where the report expects prose.

2. Degradation. Every failure path must return `("", [], "")` rather than raise:
   `Scraper.extract_data_from_url` has no per-URL recovery, so one raising scraper
   loses every other URL in the same batch.

These run offline against captured payloads. The live path is exercised by the
container itself -- `docker exec gptr-mcp-server python -c "...Crawl4AIScraper..."` --
because a unit test that needs Chromium is not a unit test.
"""
from unittest import mock

import pytest
import requests

from gpt_researcher.scraper import Crawl4AIScraper
from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.utils.workers import WorkerPool

# Trimmed from a real 0.9.3 response to https://example.com, captured 2026-09-20.
REAL_ENVELOPE = {
    "success": True,
    "results": [{
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "cleaned_html": "<div>Example Domain</div>",
        "markdown": {
            "raw_markdown": "# Example Domain\n\nThis domain is for use in examples.",
            "markdown_with_citations": "# Example Domain [1]",
            "references_markdown": "[1] https://iana.org",
            "fit_markdown": "",          # empty unless a content filter is configured
            "fit_html": "",
        },
        "metadata": {"title": "Example Domain", "description": "..."},
        "media": {"images": [{"src": "https://example.com/a.png", "score": 3},
                             {"alt": "no src here"}],
                  "videos": [], "audio": []},
    }],
}


def _scraper():
    """A Scraper wired to this backend. The worker pool is a constructor requirement,
    not something get_scraper consults."""
    return Scraper(["https://example.com"], "", "crawl4ai", WorkerPool(1))


def _response(payload, status=200):
    reply = mock.Mock(spec=requests.Response)
    reply.status_code = status
    reply.json.return_value = payload
    reply.raise_for_status.return_value = None
    return reply


def _scrape(payload=None, exc=None):
    with mock.patch.object(requests, "post",
                           side_effect=exc or None,
                           return_value=None if exc else _response(payload)):
        return Crawl4AIScraper("https://example.com", requests.Session()).scrape()


# ------------------------------------------------------------------ the envelope

def test_the_real_payload_yields_prose_a_title_and_images():
    content, images, title = _scrape(REAL_ENVELOPE)

    assert content.startswith("# Example Domain"), (
        f"content came back as {content[:60]!r} -- 0.9.3 nests the text under "
        "markdown.raw_markdown, and reading markdown directly hands the report a dict")
    assert title == "Example Domain", f"title was {title!r}"
    assert images == ["https://example.com/a.png"], (
        f"images were {images!r}; entries without a src are decorative and must drop out")


def test_a_filtered_server_is_preferred_over_the_raw_text():
    """`fit_markdown` is the boilerplate-stripped variant. It is empty by default, which
    is why raw is the fallback and not the other way round -- but when a content filter
    IS configured, using raw would silently keep the navigation the filter removed."""
    payload = {"results": [dict(REAL_ENVELOPE["results"][0])]}
    payload["results"][0]["markdown"] = {"raw_markdown": "nav nav nav ARTICLE",
                                         "fit_markdown": "ARTICLE"}
    content, _, _ = _scrape(payload)

    assert content == "ARTICLE", f"got {content!r} -- the filtered text must win when present"


def test_a_plain_string_markdown_still_reads():
    """Older servers and the /md endpoint send markdown as a bare string."""
    content, _, _ = _scrape({"results": [{"markdown": "# Plain", "success": True}]})
    assert content == "# Plain", f"got {content!r}"


# ----------------------------------------------------------------- degradation

@pytest.mark.parametrize("name,payload,exc", [
    ("connection refused", None, requests.ConnectionError("refused")),
    ("timeout", None, requests.Timeout("too slow")),
    ("empty results", {"success": True, "results": []}, None),
    ("render failed", {"results": [{"success": False,
                                    "error_message": "net::ERR_NAME_NOT_RESOLVED",
                                    "markdown": {"raw_markdown": "Cloudflare says no"}}]}, None),
    ("garbage", "not json at all", None),
])
def test_every_failure_returns_empty_instead_of_raising(name, payload, exc):
    """`success: false` still arrives as HTTP 200 with a body. Returning that body puts
    an error page into the report as though it were the source."""
    assert _scrape(payload, exc) == ("", [], ""), f"{name} did not degrade quietly"


def test_the_token_is_sent_when_one_is_configured(monkeypatch):
    """0.9.3 binds the container's own loopback unless CRAWL4AI_API_TOKEN is set, and
    setting it turns Bearer auth on for every endpoint -- so a reachable server is
    always an authenticated one."""
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", "t0ken")
    with mock.patch.object(requests, "post", return_value=_response(REAL_ENVELOPE)) as post:
        Crawl4AIScraper("https://example.com", requests.Session()).scrape()

    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer t0ken"}, (
        f"sent {post.call_args.kwargs.get('headers')!r}; without the bearer header every "
        "request to a token-enabled server is a 401")


# -------------------------------------------------------------------- wiring

def test_the_scraper_is_reachable_by_name():
    """SCRAPER=crawl4ai in the compose file selects this class. An unregistered name
    raises 'Scraper not found.' at the first URL of the first run, not at boot."""
    assert _scraper().get_scraper(
        "https://example.com/page") is Crawl4AIScraper


def test_pdfs_and_arxiv_abstracts_still_bypass_it():
    """URL-shape routing sits above the configured scraper: a PDF handed to Chromium
    comes back as a viewer shell, not the paper."""
    scraper = _scraper()
    assert scraper.get_scraper("https://x.com/paper.pdf").__name__ == "PyMuPDFScraper"
    assert scraper.get_scraper("https://arxiv.org/abs/2509.16198").__name__ == "ArxivScraper"
