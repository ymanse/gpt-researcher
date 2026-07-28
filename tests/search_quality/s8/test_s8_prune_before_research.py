"""s8: a node destined for PRUNED must not be researched first, and the expander
must be able to see what is already queued.

Both behaviours come from the pending-node RCA (harness-search/no_read/audit/
pending_rca.md), measured on benchmark round 4:

  - 23-62% of everything the tree researched was pruned immediately afterwards.
    compute_novelty reads only the node's QUESTION embedding, so that search, its
    scrapes and an answer LLM call were bought and thrown away.
  - `_covered_ground()` omits PENDING, so the expansion prompt could not tell the
    model that the question it was inventing was already sitting in the frontier.
    Siblings therefore researched near-identical questions by construction — the
    upstream source of the roll-up redundancy the s7 gate measures.

Deterministic: no network, no real LLM, no embeddings service.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.tree_research as tr


ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
NEAR_Q = "Which practical failure modes does the transactional outbox pattern show?"
FAR_Q = "Which vendors ship a managed change-data-capture connector?"

# cos(ROOT, NEAR) == 0.8 -> novelty 0.20, under the 0.30 floor => NEAR must be pruned.
# Still below DEDUP_COSINE (0.92), so it is really created, exactly as the s4 contract
# requires — what changes here is only that it is pruned BEFORE paying for research.
EMB = {
    ROOT_Q: [1.0, 0.0, 0.0],
    NEAR_Q: [0.8, 0.6, 0.0],
    FAR_Q: [0.0, 0.0, 1.0],
}


def _skill() -> tr.TreeResearchSkill:
    """Same parent shape the s4 tests use — TreeResearchSkill reads tone/websocket/
    headers/visited_urls off it in __init__."""
    parent = SimpleNamespace(
        query=ROOT_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    return tr.TreeResearchSkill(parent)


def _node(nid: str, q: str, depth: int = 1, status=tr.NodeStatus.PENDING) -> tr.ResearchNode:
    n = tr.ResearchNode(id=nid, question=q, parent_id=None, depth=depth)
    n.status = status
    n.question_embedding = EMB.get(q)
    return n


def test_pruned_node_is_never_researched():
    """The whole point: identifying a duplicate must not cost a research pass."""
    skill = _skill()
    root = _node("0", ROOT_Q, depth=0)
    near = _node("0.0", NEAR_Q)
    skill.nodes = {"0": root, "0.0": near}
    # root's question counts as ground already covered
    skill._covered_embeddings = [EMB[ROOT_Q]]

    researched: list[str] = []

    async def _research(node):
        researched.append(node.id)
        node.status = tr.NodeStatus.ANSWERED

    with mock.patch.object(skill, "research_node", side_effect=_research):
        novelty = skill.compute_novelty(near)
        assert novelty < 0.30, f"the near-duplicate must score under the floor, got {novelty}"

    assert researched == [], (
        "scoring novelty must not require researching the node — it reads the question "
        "embedding only, which is what makes prune-before-research possible"
    )


def test_queued_questions_are_shown_to_the_expander_but_not_as_covered_ground():
    """PENDING questions must reach the expansion prompt, and must NOT be presented
    as ground already covered — most pending nodes are never researched, so calling
    them covered would fence expansion away from a hole nobody filled (the same
    reason FAILED nodes are excluded)."""
    skill = _skill()
    answered = _node("0", ROOT_Q, depth=0, status=tr.NodeStatus.EXPANDED)
    answered.answer_digest = "Outbox tables grow without bound unless cleanup is built in."
    queued = _node("0.1", FAR_Q)
    skill.nodes = {"0": answered, "0.1": queued}

    covered = "\n".join(skill._covered_ground())
    assert ROOT_Q in covered, "a researched question is covered ground"
    assert FAR_Q not in covered, (
        "a QUEUED question is not covered ground — presenting it as covered would stop "
        "the tree from ever filling that hole if the node is never researched"
    )

    queued_block = skill._queued_ground()
    assert FAR_Q in queued_block, "the expander must see what is already in the frontier"
    assert "QUEUED" in queued_block
    assert "do NOT restate" in queued_block


def test_queued_block_is_empty_when_nothing_is_pending():
    skill = _skill()
    skill.nodes = {"0": _node("0", ROOT_Q, depth=0, status=tr.NodeStatus.EXPANDED)}
    assert skill._queued_ground() == "", (
        "no pending nodes must add no section, so the prompt does not carry an empty heading"
    )


def test_failed_node_registration_is_withdrawn():
    """Novelty is scored (and the question registered) before research, so a node that
    then FAILS must have its registration removed — otherwise the starved question
    counts as covered ground and prunes the later child that would have filled it."""
    skill = _skill()
    node = _node("0.2", FAR_Q)
    before = len(skill._covered_embeddings)
    skill.compute_novelty(node)
    assert len(skill._covered_embeddings) == before + 1, "scoring registers the question"
    skill.unregister_covered(node)
    assert len(skill._covered_embeddings) == before, (
        "a FAILED node researched nothing, so its question must not stay covered ground"
    )


@pytest.mark.asyncio
async def test_run_prunes_without_researching_the_duplicate(tmp_path):
    """End to end through run(): the near-duplicate lands as PRUNED and counts towards
    pruned_count, but research_node is never invoked for it."""
    skill = _skill()
    researched: list[str] = []

    async def _research(node):
        researched.append(node.question)
        node.status = tr.NodeStatus.ANSWERED
        node.answer_md = "answer body"
        node.answer_digest = "answer body"
        node.learnings = ["answer body"]

    async def _embed(text):
        # the seam the s4 tests use — skill.embed_question, no embeddings service
        return list(EMB.get(text, [0.0, 0.0, 0.1]))

    async def _children(node):
        return [NEAR_Q, FAR_Q] if node.depth == 0 else []

    skill.embed_question = _embed
    with mock.patch.object(skill, "research_node", side_effect=_research), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children), \
         mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value="ANSWER: x\nDIGEST: x\nLEARNINGS:\n- x\n")):
        result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2, max_nodes=10,
                                 outputs_dir=str(tmp_path))

    assert NEAR_Q not in researched, (
        "the near-duplicate was researched before being pruned — that is exactly the "
        "wasted search/scrape/LLM pass this change removes"
    )
    assert FAR_Q in researched, "the genuinely novel sibling must still be researched"
    # result["tree"]["nodes"] is keyed by node id, so read the values
    pruned = [n["status"] for n in result["tree"]["nodes"].values() if n["question"] == NEAR_Q]
    assert pruned == ["pruned"], (
        f"the duplicate must still be CREATED and land as PRUNED (s4 contract), got {pruned}"
    )
    assert result["tree"]["meta"]["pruned_count"] > 0
