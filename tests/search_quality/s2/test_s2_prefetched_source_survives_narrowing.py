"""Regression guard — review R1: a prefetched source must be able to survive narrowing.

Two halves of one causal chain, each pinned here:

1. _search_relevant_source_urls' prefetch branch (Firecrawl / PubMed Central) never
   routed its URL through _get_new_urls, the only path that adds a URL to
   visited_urls — so a document the retriever had already fetched in full was
   missing from the researcher's own visited set (and from report references).
2. research_node built node.sources exclusively from visited_urls, so such a
   document was excluded upstream of the "read AND quoted" filter, unconditionally,
   however faithfully the node answer quoted it. Candidates now come from the
   documents actually read (research_sources).

Deterministic, no network.
"""
from types import SimpleNamespace

URL_PREFETCHED = "https://prefetched.example.com/doc"
URL_NEEDS_SCRAPING = "https://needs-scraping.example.com/doc"
URL_RETRIEVER_ONLY = "https://retriever-only.example.com/doc"

QUOTED_SENTENCE = "Photosystem II oxidizes water at a manganese-calcium cluster in 2024"
PREFETCHED_DOC = (
    "Preamble prose the answer never uses. "
    f"{QUOTED_SENTENCE}. Trailing prose the answer never uses."
)

LLM_RESPONSE = (
    f"ANSWER: {QUOTED_SENTENCE}.\n"
    f"DIGEST: {QUOTED_SENTENCE}.\n"
    "LEARNINGS:\n"
    f"- {QUOTED_SENTENCE}\n"
)


class _PrefetchRetriever:
    def __init__(self, query, query_domains=None):
        self.query = query

    def search(self, max_results=10):
        return [
            {"href": URL_PREFETCHED, "raw_content": "Full prefetched body. " * 30},
            {"href": URL_NEEDS_SCRAPING, "body": "short snippet only"},
        ]


async def test_prefetched_url_is_marked_visited():
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

    new_urls, prefetched, _ = await ResearchConductor(
        researcher)._search_relevant_source_urls("q")

    # dedup/scraping behaviour unchanged: prefetched results are never re-scraped
    assert new_urls == [URL_NEEDS_SCRAPING]
    assert [p["url"] for p in prefetched] == [URL_PREFETCHED]
    assert URL_PREFETCHED in researcher.visited_urls, (
        "a document the retriever already fetched in full was read — it must be "
        "recorded as visited, or it is invisible to report references and to "
        "every consumer that reads visited_urls"
    )


async def test_read_and_quoted_source_survives_even_if_never_visited(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            # the prefetched URL is deliberately absent: node.sources must not
            # depend on visited_urls membership for its candidates
            self.visited_urls.add(URL_RETRIEVER_ONLY)
            return "collected context"

        def get_research_sources(self):
            return [{"url": URL_PREFETCHED, "raw_content": PREFETCHED_DOC}]

        def get_costs(self):
            return 0.0

    async def fake_chat(*args, **kwargs):
        return LLM_RESPONSE

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    skill = tree_mod.TreeResearchSkill(SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    ))
    node = tree_mod.ResearchNode(id="0", question="root question",
                                 parent_id=None, depth=0)
    await skill.research_node(node)

    assert node.sources == [URL_PREFETCHED], (
        "a read-and-quoted document must survive narrowing regardless of how it "
        "was obtained; a retriever-only URL must still not survive"
    )
