"""Stage 3 tests — the citation pass must verify against pages already read.

Measured on this deployment (gptr-mcp-server, 2026-07-29 deep_research run):
the run finished its last sub-research at 06:48:41 and did not return until
07:01:44 — 13 of 37 minutes inside CitationAgent.verify. DeepResearchSkill
called it as `verify(results['citations'])`, dropping the `documents` seed,
so all 66 claims were re-fetched one at a time via firecrawl /v2/scrape with
maxAge=0 (cache bypass) — pages the same run had just scraped and still held
in researcher.research_sources (measured: 126 unique urls, median 17KB of
raw_content each).

The contract these tests pin: everything the run already read is verified in
memory; only a url we never scraped may reach the network.

Deterministic, no network: requests.post is replaced by a tripwire.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

QUOTE = "the grid stores excess solar energy in batteries during peak production"
PAGE = (
    "# Grid storage\n\nUtilities report that the grid stores excess solar energy "
    "in batteries during peak production hours.\n"
)


@pytest.fixture(autouse=True)
def firecrawl_key(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")


def _skill(sources):
    """DeepResearchSkill over a stub researcher carrying already-scraped sources."""
    from gpt_researcher.skills.deep_research import DeepResearchSkill

    researcher = SimpleNamespace(
        cfg=SimpleNamespace(config_path=None),
        websocket=None,
        tone=None,
        headers={},
        visited_urls=set(),
        research_sources=sources,
        get_research_sources=lambda: sources,
    )
    return DeepResearchSkill(researcher)


@pytest.mark.asyncio
async def test_a_cited_page_the_run_already_scraped_is_not_refetched():
    skill = _skill([{"url": "https://a.example.com/grid", "raw_content": PAGE}])

    with mock.patch("requests.post", side_effect=AssertionError("re-scraped a page already in hand")):
        verification = await skill.verify_citations({QUOTE: "https://a.example.com/grid"})

    assert verification["total_claims"] == 1
    assert verification["grounded"] == 1


@pytest.mark.asyncio
async def test_a_url_never_scraped_still_reaches_the_network():
    """The seed is a cache, not a wall — an uncited-source url must still be checked."""
    skill = _skill([{"url": "https://a.example.com/grid", "raw_content": PAGE}])

    resp = mock.MagicMock(status_code=200)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"success": True, "data": {"markdown": PAGE}}

    with mock.patch("requests.post", return_value=resp) as post:
        verification = await skill.verify_citations({QUOTE: "https://elsewhere.example.com/x"})

    assert post.call_count == 1
    assert verification["total_claims"] == 1


@pytest.mark.asyncio
async def test_sources_without_content_do_not_poison_the_seed():
    """An empty raw_content must not register as 'we have this page' — that would
    silently mark every claim citing it unverified instead of fetching it."""
    skill = _skill([{"url": "https://a.example.com/grid", "raw_content": ""}])

    resp = mock.MagicMock(status_code=200)
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"success": True, "data": {"markdown": PAGE}}

    with mock.patch("requests.post", return_value=resp) as post:
        verification = await skill.verify_citations({QUOTE: "https://a.example.com/grid"})

    assert post.call_count == 1
    assert verification["grounded"] == 1
