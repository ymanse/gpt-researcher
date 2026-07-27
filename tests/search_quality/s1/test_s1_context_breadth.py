"""s1 tests — defect 2 (context starvation), the two places context is throttled.

Both failures were measured in bench round 2, not guessed:

  (a) the frontier was researched strictly sequentially (~45s a node), so
      time_budget_s — not max_nodes — decided how much of the tree ever became
      context: denorm-derived-table ended 12 of 21 nodes PENDING, outbox 17 of 29,
      and the primary-source questions those nodes carried were never asked.

  (b) the per-sub-query context window keeps a flat top-N prefix of the chunks
      EmbeddingsFilter ranked, so one long page whose wording echoes the query can
      occupy the whole window while every other scraped page contributes nothing.

Deterministic: no network, no LLM, no wall-clock assertions.
"""
import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document

import gpt_researcher.skills.tree_research as tree_mod
from gpt_researcher.context.compression import spread_across_sources


# ---------------------------------------------------------------------------
# (a) the frontier researches a batch of nodes concurrently
# ---------------------------------------------------------------------------

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
CHILD_QS = [f"What does source {i}'s own documentation report?" for i in range(3)]


def _parent_stub(query):
    cfg = SimpleNamespace(strategic_llm_provider="p", strategic_llm_model="m",
                          fast_llm_provider="p", fast_llm_model="m")
    return SimpleNamespace(query=query, cfg=cfg, tone=None, websocket=None,
                           headers={}, visited_urls=set())


async def test_frontier_researches_a_batch_of_nodes_concurrently():
    """The three children must be in flight at the same time. Sequential research
    peaks at one node live, which is the observed defect: the clock, not the node
    budget, is what cut every measured tree short."""
    live = 0
    peak = 0

    async def fake_research(node):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)  # yield: a sequential caller can never overlap here
        live -= 1
        node.answer_md = f"Finding for {node.id}."
        node.answer_digest = node.answer_md
        node.learnings = [node.answer_md]
        node.sources = []
        node.status = tree_mod.NodeStatus.ANSWERED

    skill = tree_mod.TreeResearchSkill(_parent_stub(ROOT_Q))
    skill.research_node = fake_research

    async def fake_children(node):
        return list(CHILD_QS) if node.depth == 0 else []

    skill.generate_child_questions = fake_children

    # orthogonal embeddings: nothing is a dedup drop and nothing prunes, so
    # scheduling is the only thing this test measures
    vectors = {ROOT_Q: [1.0, 0.0, 0.0, 0.0]}
    for i, q in enumerate(CHILD_QS):
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[i + 1] = 1.0
        vectors[q] = vec

    async def fake_embed(text):
        return list(vectors.get(text, [0.0, 0.0, 0.0, 1.0]))

    skill.embed_question = fake_embed

    await skill.run(query=ROOT_Q, max_depth=1, max_breadth=3, max_nodes=4,
                    node_concurrency=3)

    assert peak >= 2, (
        "the frontier must research a batch of nodes concurrently — peaking at "
        f"{peak} node(s) live means the tree is still researched one node at a "
        "time, which is what left 12 of 21 (denorm) and 17 of 29 (outbox) nodes "
        "PENDING when time_budget_s expired"
    )


async def test_batch_never_researches_past_max_nodes():
    """Concurrency must not buy extra nodes: a budget of 2 researches the root plus
    exactly ONE child, whatever the batch size."""
    researched = []

    async def fake_research(node):
        researched.append(node.id)
        node.answer_md = f"Finding for {node.id}."
        node.answer_digest = node.answer_md
        node.learnings = [node.answer_md]
        node.sources = []
        node.status = tree_mod.NodeStatus.ANSWERED

    skill = tree_mod.TreeResearchSkill(_parent_stub(ROOT_Q))
    skill.research_node = fake_research

    async def fake_children(node):
        return list(CHILD_QS) if node.depth == 0 else []

    skill.generate_child_questions = fake_children

    counter = iter(range(1, 99))

    async def fake_embed(text):
        vec = [0.0] * 8
        vec[next(counter) % 8] = 1.0
        return vec

    skill.embed_question = fake_embed

    await skill.run(query=ROOT_Q, max_depth=1, max_breadth=3, max_nodes=2,
                    node_concurrency=3)

    assert len(researched) == 2, (
        "max_nodes bounds the batch too — a concurrent pop must never spend more "
        f"of the node budget than the sequential one did. researched {researched}"
    )


# ---------------------------------------------------------------------------
# (b) the retained context window spans the documents actually read
# ---------------------------------------------------------------------------

def _chunk(source: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "title": source})


def test_retained_chunks_spread_across_sources_before_a_second_from_one():
    """One page ranking every chunk highest must not take the whole window: the
    postgres/quantumscape/nvidia sentence a golden fact needs sits on a page whose
    single best chunk ranked below that page's ninth."""
    ranked = [_chunk("https://hog.example/page", f"hog chunk {i}") for i in range(10)]
    ranked += [_chunk("https://spec.example/doc", "the spec sentence"),
               _chunk("https://vendor.example/spec", "the vendor figure")]

    spread = spread_across_sources(ranked)
    window = spread[:3]
    sources = [d.metadata["source"] for d in window]

    assert len(set(sources)) == 3, (
        "each source's best chunk must come before any source's second — a "
        f"top-N window over {sources} still starves every page but the first"
    )
    assert [d.page_content for d in window[1:]] == ["the spec sentence",
                                                    "the vendor figure"]


def test_spread_preserves_relevance_order_within_and_between_sources():
    ranked = [_chunk("a", "a1"), _chunk("b", "b1"), _chunk("a", "a2"), _chunk("b", "b2")]

    assert [d.page_content for d in spread_across_sources(ranked)] == \
        ["a1", "b1", "a2", "b2"]


def test_spread_of_nothing_is_nothing():
    assert spread_across_sources([]) == []
