"""RED tests for s4 — coverage-driven expansion + embedding novelty (spec/search-quality.md,
observed defects 4 and 5).

Observed:
  (4) generate_child_questions tells the model which QUESTIONS already exist in the tree
      ("do NOT overlap them") but never what those nodes' research actually FOUND
      (answer_digest / learnings). A model steering away from repeat wording has no
      signal about which TOPICS are already covered, so a differently-worded child can
      silently re-cover an already-answered area.
  (5) compute_novelty measures uniqueness of node.learnings TEXT (exact-ish string
      match against a seen-strings set), not of the underlying MEANING. A child whose
      answer restates an already-covered finding in different words is scored fully
      novel, so pruned_count stays 0 in every observed live tree (bun-rust-port: 13
      nodes, 0 pruned) even though the whole point of pruning is to catch exactly this.

Contract pinned here (implemented in s4-impl):
  (a) generate_child_questions's prompt to the model must surface OTHER nodes' actual
      findings (answer_digest/learnings), not just their question text, so the model
      can steer toward genuinely uncovered topics.
  (b) compute_novelty compares question EMBEDDINGS (cosine), not learnings text: two
      nodes whose questions mean the same thing but are worded completely differently
      must score low novelty.
  (c) end-to-end in run(): a child whose computed novelty falls below novelty_threshold
      is marked PRUNED and is never expanded (no further children, no further research).

Deterministic, no network: GPTResearcher and create_chat_completion are patched at the
tree_research module seam; embed_question is replaced on the instance with a fixed
lookup table so every embedding in a test is exact and reproducible.
"""
import math
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod


def _parent_stub(query: str) -> SimpleNamespace:
    return SimpleNamespace(
        query=query,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )


def _unit_vec(cosine_to_x_axis: float) -> list[float]:
    """A 2D unit vector whose cosine similarity to (1.0, 0.0) is exactly `cosine_to_x_axis`."""
    return [cosine_to_x_axis, math.sqrt(1.0 - cosine_to_x_axis ** 2)]


# ---------------------------------------------------------------------------
# (a) expansion prompt must surface other nodes' FINDINGS, not just their questions
# ---------------------------------------------------------------------------

COVERED_TOPIC_FACT = (
    "Sharding split the port across 4 worktrees running 16 Claude agents each, "
    "64 parallel workers total."
)


async def test_expansion_prompt_reflects_other_nodes_covered_findings(monkeypatch):
    prompts = []

    async def fake_chat(messages=None, **kwargs):
        prompts.append(" ".join(str(m.get("content", "")) for m in (messages or [])))
        return "Question: some follow-up\n"

    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)
    skill = tree_mod.TreeResearchSkill(_parent_stub("root question"))

    sibling = tree_mod.ResearchNode(id="0", question="How was the work sharded?",
                                     parent_id=None, depth=0)
    sibling.status = tree_mod.NodeStatus.ANSWERED
    sibling.answer_digest = COVERED_TOPIC_FACT
    sibling.learnings = [COVERED_TOPIC_FACT]
    skill.nodes[sibling.id] = sibling

    target = tree_mod.ResearchNode(id="1", question="What else is notable about the port?",
                                   parent_id=None, depth=0)
    target.status = tree_mod.NodeStatus.ANSWERED
    target.answer_digest = "An unrelated digest about validation layers and fuzzing."
    skill.nodes[target.id] = target

    await skill.generate_child_questions(target)

    assert prompts, "generate_child_questions must call the LLM"
    assert COVERED_TOPIC_FACT in prompts[-1], (
        "observed defect 4: the expansion prompt lists only OTHER nodes' raw question "
        "text ('existing questions ... do NOT overlap them'), never what those nodes "
        "actually found (answer_digest/learnings) -- the model has no way to see which "
        "TOPICS are already covered, only which literal phrasings to avoid, so a "
        "differently-worded child can silently re-cover an already-answered area"
    )


# ---------------------------------------------------------------------------
# (b) compute_novelty is embedding-cosine based, not learnings-text based
# ---------------------------------------------------------------------------

def test_compute_novelty_catches_reworded_duplicate_via_embedding():
    skill = tree_mod.TreeResearchSkill(_parent_stub("root question"))

    seen = tree_mod.ResearchNode(id="0", question="How was the codebase sharded across workers?",
                                  parent_id=None, depth=0)
    seen.learnings = ["The port split work across 4 worktrees with 16 agents each."]
    seen.question_embedding = _unit_vec(1.0)
    first_novelty = skill.compute_novelty(seen)
    assert first_novelty >= 0.30, "the first-ever node must not itself be starved of novelty"

    # completely different wording, essentially the same underlying question --
    # only the embedding says so, the text shares almost no tokens with `seen`
    paraphrase = tree_mod.ResearchNode(
        id="1", question="Across how many parallel worker processes was the migration divided?",
        parent_id=None, depth=1)
    paraphrase.learnings = ["Migration effort spread over sixteen agents times four teams."]
    paraphrase.question_embedding = _unit_vec(0.99)

    novelty = skill.compute_novelty(paraphrase)

    assert novelty < 0.30, (
        "observed defect 5: compute_novelty matches learnings TEXT, so a semantically "
        "near-duplicate child in completely different wording scores fully novel; it "
        "must compare question embeddings (cosine) and recognize the near-duplicate "
        "regardless of wording"
    )


def test_failed_sibling_embedding_does_not_leak_into_novelty_pool():
    """A FAILED node's question_embedding is assigned at child-creation time,
    before research ever runs, so it still sits on the node object (and in
    skill._embeddings) even though compute_novelty is never called for the
    FAILED node itself. If that embedding is still visible to OTHER nodes'
    novelty scoring, a topically-similar sibling that starved on a retriever
    failure can prune away a genuinely-researched node that would have filled
    the hole it left."""
    skill = tree_mod.TreeResearchSkill(_parent_stub("root question"))

    root = tree_mod.ResearchNode(id="0", question="root question", parent_id=None, depth=0)
    root.status = tree_mod.NodeStatus.EXPANDED
    root.question_embedding = [1.0, 0.0, 0.0]
    skill.nodes[root.id] = root
    skill._embeddings.append(list(root.question_embedding))

    failed = tree_mod.ResearchNode(id="0.0", question="starved question", parent_id="0", depth=1)
    failed.status = tree_mod.NodeStatus.FAILED
    failed.question_embedding = [0.0, 1.0, 0.0]
    skill.nodes[failed.id] = failed
    skill._embeddings.append(list(failed.question_embedding))

    real = tree_mod.ResearchNode(id="0.1", question="REAL FINDING that must not be pruned away",
                                  parent_id="0", depth=1)
    real.status = tree_mod.NodeStatus.ANSWERED
    # cosine(real, failed) == 0.80 (well past the 0.70 similarity novelty_threshold
    # treats as redundant); cosine(real, root) == 0.0
    real.question_embedding = [0.0, 0.8, 0.6]

    novelty = skill.compute_novelty(real)

    assert novelty >= 0.30, (
        "a FAILED sibling's leaked embedding (cosine 0.8) would score this "
        "genuinely new, already-researched question as a near-duplicate and "
        f"prune it away, even though the FAILED node never covered anything. "
        f"got {novelty}"
    )


# ---------------------------------------------------------------------------
# (c) end-to-end: a low-novelty child is marked PRUNED and never expanded
# ---------------------------------------------------------------------------

ROOT_Q = "How was the Bun Rust port executed?"
CHILD_Q_1 = "How was the codebase sharded across workers?"
CHILD_Q_2 = "Across how many parallel worker processes was the port divided?"

ROOT_ANSWER = "The port used a Claude Code agent harness across many parallel workers."
CHILD_1_ANSWER = "The work was split across 4 worktrees running 16 agents each."
CHILD_2_ANSWER = "Sixteen agents times four teams handled the parallel migration work."

# cosine(CHILD_Q_1, CHILD_Q_2) embeddings == 0.85: close enough for low novelty (< the
# default 0.30 threshold, however novelty is derived from cosine) but comfortably below
# DEDUP_COSINE (0.92), so child 2 must still become a real node, not be dropped as an
# exact-duplicate candidate before it is ever researched
EMB = {
    CHILD_Q_1: _unit_vec(1.0),
    CHILD_Q_2: _unit_vec(0.85),
}

RICH_CONTEXT = "Collected page text from the scraped sources. " * 1200  # >> MIN_CONTEXT_CHARS
DOC_BY_Q = {
    ROOT_Q: "Engineering blog. " + ROOT_ANSWER + " Further unrelated archive prose. " * 20,
    CHILD_Q_1: "Engineering blog. " + CHILD_1_ANSWER + " Further unrelated archive prose. " * 20,
    CHILD_Q_2: "Engineering blog. " + CHILD_2_ANSWER + " Further unrelated archive prose. " * 20,
}
ANSWER_BY_Q = {ROOT_Q: ROOT_ANSWER, CHILD_Q_1: CHILD_1_ANSWER, CHILD_Q_2: CHILD_2_ANSWER}


def _llm_response(answer: str) -> str:
    return f"ANSWER: {answer}\nDIGEST: {answer}\nLEARNINGS:\n- {answer}\n"


async def test_low_novelty_child_is_pruned_and_never_expanded(monkeypatch):
    url_by_q = {q: f"https://example.com/{i}" for i, q in enumerate(DOC_BY_Q)}

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            self.visited_urls.add(url_by_q[self.query])
            return RICH_CONTEXT

        def get_research_sources(self):
            return [{"url": url_by_q[self.query], "title": "doc",
                     "raw_content": DOC_BY_Q[self.query]}]

        def get_costs(self):
            return 0.0

    async def fake_chat(messages=None, **kwargs):
        content = " ".join(str(m.get("content", "")) for m in (messages or []))
        if "Generate up to" in content:
            # root's expansion: propose both children in one shot
            return f"Question: {CHILD_Q_1}\nQuestion: {CHILD_Q_2}\n"
        for q, answer in ANSWER_BY_Q.items():
            if q in content:
                return _llm_response(answer)
        return _llm_response("unmapped")

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    skill = tree_mod.TreeResearchSkill(_parent_stub(ROOT_Q))

    async def fake_embed(text):
        for q, emb in EMB.items():
            if q in text:
                return emb
        return [1.0, 0.0]

    skill.embed_question = fake_embed

    result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2, max_nodes=5)

    nodes = result["tree"]["nodes"]
    assert nodes["0"]["status"] in ("answered", "expanded"), (
        "no regression: the root must research normally and expand"
    )
    child_ids = nodes["0"]["children"]
    assert len(child_ids) == 2, (
        "both candidate children must become real nodes -- their embedding cosine "
        "(0.85) is below DEDUP_COSINE (0.92), so neither is dropped as an exact "
        "duplicate at generation time; only novelty-based pruning (this test's subject) "
        "may remove the second one"
    )

    statuses = {cid: nodes[cid]["status"] for cid in child_ids}
    pruned = [cid for cid, st in statuses.items() if st == "pruned"]
    assert pruned, (
        "observed defect 5 in the end-to-end wiring: pruned_count is ALWAYS 0 in "
        "every live tree because compute_novelty never recognizes a reworded "
        "duplicate; the second child here restates the first child's finding in "
        "completely different words with a 0.85-cosine question embedding, and must "
        "be marked PRUNED"
    )

    pruned_id = pruned[0]
    assert nodes[pruned_id]["children"] == [], (
        "a PRUNED node must never be expanded further -- no grandchildren"
    )
