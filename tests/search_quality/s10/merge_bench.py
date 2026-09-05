"""s10 rig — measures how much of what the retrievers found survives the merge.

Two places lose material between "the retrievers returned a url" and "the report
LLM sees the text", and neither is visible in any existing metric:

  (a) claim, not dedup. The sub-queries of one pass run concurrently
      (asyncio.gather in _get_context_by_web_search) and the first one to reach
      _get_new_urls takes the url. visited_urls then reads as "only the first
      sub-query may read this page", so every sibling that found the same page
      compresses without it — against the leftovers, or, when its whole result set
      was claimed, against nothing.

  (b) ` `.join(). The per-sub-query contexts were concatenated: a page several
      sub-queries retained was paid for once per sub-query, and whatever clips the
      context downstream keeps a prefix, so a long early sub-query can eat the
      whole budget before a later one contributes a line.

Deterministic: no network, no LLM, no embeddings, no wall-clock. Fake retrievers
hand out an overlapping url set per sub-query, a fake scraper serves fixed page
bodies. Pages are small enough to take ContextCompressor's fast path — the merge
being measured is identical on both compression paths (chunking is deterministic
per document, so a chunk two sub-queries both retain is byte-identical either
way), so skipping embeddings costs no coverage.

    python tests/search_quality/s10/merge_bench.py

Metrics
  sub_query_source_coverage  served (sub_query, source) pairs / retrieved pairs.
                             1.0 = every sub-query saw every page its own
                             retrievers found. Order-independent: under (a) each
                             url reaches exactly one sub-query whichever one wins
                             the race, so the ratio is stable even though the
                             winner is not.
  starved_sub_queries        sub-queries whose context came back empty.
  duplicate_blocks           blocks in the final context past the first with the
                             same (source, content).
  scrape_calls               urls actually fetched. Restoring coverage by
                             re-fetching is not a fix — this must stay at the
                             number of distinct pages.
  sources_in_context         distinct sources reaching the final context.
  context_chars              final merged length.
"""
import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gpt_researcher.prompts import PromptFamily  # noqa: E402
from gpt_researcher.skills.context_manager import ContextManager  # noqa: E402
from gpt_researcher.skills.researcher import ResearchConductor  # noqa: E402

QUERY = "What breaks when a transactional outbox is replayed?"

# Overlapping on purpose — sub-queries of one pass are rephrasings of one question
# and their retrievers land on the same pages. sq_b's whole result set is claimable
# by its siblings, which is the starvation case.
SUB_QUERIES = [
    "outbox replay duplicate delivery",
    "outbox replay ordering guarantees",
    "outbox relay crash recovery",
]
RETRIEVED = {
    SUB_QUERIES[0]: ["https://a.example/dup", "https://b.example/idem", "https://c.example/order"],
    SUB_QUERIES[1]: ["https://b.example/idem", "https://c.example/order"],
    SUB_QUERIES[2]: ["https://c.example/order", "https://d.example/relay"],
    QUERY: ["https://a.example/dup", "https://e.example/outbox"],
}
PAGES = {
    "https://a.example/dup": "At-least-once delivery means a replayed outbox row is delivered twice.",
    "https://b.example/idem": "Consumers must be idempotent; dedupe on the message id, not the payload.",
    "https://c.example/order": "Per-aggregate ordering survives replay only if the relay reads by sequence.",
    "https://d.example/relay": "A relay that crashes after publish but before marking sent republishes on restart.",
    "https://e.example/outbox": "The outbox table is written in the same transaction as the aggregate.",
}

_BLOCK_SPLIT = re.compile(r"(?m)^(?=Source: )")


def _blocks(context: str) -> list[tuple[str, str]]:
    """(source, normalized content) for each pretty_print_docs block."""
    out = []
    for block in _BLOCK_SPLIT.split(context or ""):
        if not block.strip():
            continue
        source = block.split("\n", 1)[0].removeprefix("Source: ").strip()
        _, _, content = block.partition("Content: ")
        out.append((source, " ".join(content.split())))
    return out


class _FakeRetriever:
    """Returns snippets only, so every url takes the scrape path."""

    def __init__(self, query, query_domains=None, **kwargs):
        self.query = query

    def search(self, max_results=10):
        return [{"href": url, "title": url, "body": "snippet"}
                for url in RETRIEVED.get(self.query, [])]


class _FakeScraper:
    def __init__(self, researcher):
        self.researcher = researcher
        self.fetched: list[str] = []

    async def browse_urls(self, urls):
        self.fetched.extend(urls)
        pages = [{"url": url, "raw_content": PAGES[url], "title": url}
                 for url in urls if url in PAGES]
        # Mirrors the real BrowserManager: scraped pages land in research_sources,
        # which is where a sibling reads them back from.
        self.researcher.add_research_sources(pages)
        return pages


def _researcher():
    cfg = SimpleNamespace(max_search_results_per_query=10)
    researcher = SimpleNamespace(
        cfg=cfg,
        retrievers=[_FakeRetriever],
        visited_urls=set(),
        research_sources=[],
        verbose=False,
        websocket=None,
        headers={},
        query_domains=[],
        query=QUERY,
        report_type="research_report",
        vector_store=None,
        memory=SimpleNamespace(get_embeddings=lambda: None),
        prompt_family=PromptFamily,
        kwargs={},
        add_costs=lambda *_: None,
    )
    researcher.add_research_sources = researcher.research_sources.extend
    researcher.context_manager = ContextManager(researcher)
    researcher.scraper_manager = _FakeScraper(researcher)
    return researcher


async def run() -> dict:
    """Run one full sub-query pass and return the metrics."""
    researcher = _researcher()
    conductor = ResearchConductor(researcher)

    async def _plan(query, query_domains=None):
        return list(SUB_QUERIES)

    conductor.plan_research = _plan

    per_sub_query: dict[str, str] = {}
    process = conductor._process_sub_query

    async def _spy(sub_query, scraped_data=None, query_domains=None):
        result = await process(sub_query, scraped_data or [], query_domains or [])
        per_sub_query[sub_query] = result or ""
        return result

    conductor._process_sub_query = _spy

    final = await conductor._get_context_by_web_search(QUERY)
    final = final if isinstance(final, str) else ""

    retrieved_pairs = sum(len(urls) for urls in RETRIEVED.values())
    served_pairs = sum(len({source for source, _ in _blocks(context)})
                       for context in per_sub_query.values())
    final_blocks = _blocks(final)

    return {
        "sub_query_source_coverage": round(served_pairs / retrieved_pairs, 3),
        "starved_sub_queries": sum(1 for c in per_sub_query.values() if not c.strip()),
        "duplicate_blocks": len(final_blocks) - len(set(final_blocks)),
        "scrape_calls": len(researcher.scraper_manager.fetched),
        "sources_in_context": len({source for source, _ in final_blocks}),
        "context_chars": len(final),
        "_per_sub_query": {q: len(_blocks(c)) for q, c in per_sub_query.items()},
    }


if __name__ == "__main__":
    metrics = asyncio.run(run())
    per_sub_query = metrics.pop("_per_sub_query")
    width = max(len(k) for k in metrics)
    for name, value in metrics.items():
        print(f"{name:<{width}}  {value}")
    print("\nblocks per sub-query:")
    for sub_query, count in per_sub_query.items():
        print(f"  {count:>2}  {sub_query}")
