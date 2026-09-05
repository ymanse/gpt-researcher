"""Regression guard for the s2 narrowing contract at its upstream source.

_search_relevant_source_urls splits retriever results into two branches: results
the retriever already fetched in full (raw_content > 100 chars — Firecrawl,
PubMed Central) and results that still need scraping. Only the scrape branch
recorded content in research_sources; the prefetch branch recorded
{"url": ...} alone, so tree_research's "read AND quoted" narrowing saw every
prefetched document as empty and dropped the source no matter how faithfully
the node answer quoted it.
"""
from types import SimpleNamespace

URL_PREFETCHED = "https://prefetched.example.com/doc"
URL_NEEDS_SCRAPING = "https://needs-scraping.example.com/doc"


class _PrefetchRetriever:
    def __init__(self, query, query_domains=None):
        self.query = query

    def search(self, max_results=10):
        return [
            {"href": URL_PREFETCHED, "raw_content": "Full prefetched body. " * 30},
            {"href": URL_NEEDS_SCRAPING, "body": "short snippet only"},
        ]


def _make_conductor():
    from gpt_researcher.skills.researcher import ResearchConductor

    researcher = SimpleNamespace(
        cfg=SimpleNamespace(max_search_results_per_query=5),
        retrievers=[_PrefetchRetriever],
        visited_urls=set(),
        research_sources=[],
        verbose=False,
        websocket=None,
    )
    researcher.add_research_sources = (
        lambda sources: researcher.research_sources.extend(sources))
    return ResearchConductor(researcher), researcher


async def test_prefetched_source_is_recorded_with_its_content():
    conductor, researcher = _make_conductor()

    new_urls, prefetched, found_any, _claimed_elsewhere = \
        await conductor._search_relevant_source_urls("q")

    assert found_any
    assert new_urls == [URL_NEEDS_SCRAPING]
    assert [p["url"] for p in prefetched] == [URL_PREFETCHED]

    recorded = {s["url"]: s for s in researcher.research_sources}
    assert URL_PREFETCHED in recorded, (
        "a retriever-prefetched document was read and must reach research_sources"
    )
    assert len(recorded[URL_PREFETCHED].get("raw_content") or "") > 100, (
        "prefetched content must be recorded alongside the URL — a contentless "
        "entry reads as 'never read' and makes the s2 read-and-quoted narrowing "
        "drop every Firecrawl/PubMed-Central source regardless of quotation"
    )
