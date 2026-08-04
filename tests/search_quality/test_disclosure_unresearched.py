"""A PENDING node is a question the tree accepted but never researched (budget/
time/node-cap cutoff). Measured live: a 29-node tree with 22 PENDING produced a
report containing the string "unexplored frontier" zero times — it read as a
complete answer while covering 24% of its own questions. The report must
disclose the gap instead of silently omitting it.

`assemble_report`'s post-order rollup already keeps a PENDING node's
"(unexplored frontier) {question}" text OUT of the scored body (the frozen
scorer matches a golden fact anywhere in the file, so reciting the question
that names a fact must not look like the tree found it) — that decision is
untouched here. What was missing is a reader-visible accounting of the gap,
appended AFTER the scored body/Citations, plus a mechanical N/M pair on
run()'s stats so a caller does not have to scrape the report text for it.
"""
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tr


def _bare_skill(query: str) -> tr.TreeResearchSkill:
    """A skill with no LLM, no researcher, no network — assemble_report reads
    only self.nodes; embed_question is stubbed dead so the merge stage falls
    back to word-overlap candidate selection instead of hitting a real provider."""
    parent = SimpleNamespace(
        query=query,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    skill = tr.TreeResearchSkill(parent)

    async def dead_embed(_text: str) -> list:
        raise RuntimeError("no embedding provider in this test")

    skill.embed_question = dead_embed
    return skill


def _add_node(skill: tr.TreeResearchSkill, node_id: str, question: str,
              status: tr.NodeStatus, answer: str = "", parent_id: str = "0",
              depth: int = 1) -> tr.ResearchNode:
    node = tr.ResearchNode(id=node_id, question=question, parent_id=parent_id, depth=depth)
    node.status = status
    node.answer_md = answer
    node.answer_digest = answer
    node.learnings = [answer] if answer else []
    skill.nodes[node_id] = node
    return node


QUERY = "How did the migration proceed?"
ANSWERED_Q = "What tooling drove the migration?"
ANSWER_TEXT = "The team used worktree-sharded agents to drive the migration."
PENDING_Q1 = "What does the standards body's own specification say?"
PENDING_Q2 = "What do the maintainers' own issue tracker discussions report?"


# --------------------------------------------------------------------------
# a run with pending nodes says so, and names them
# --------------------------------------------------------------------------

async def test_report_discloses_pending_questions_as_open_not_findings():
    skill = _bare_skill(QUERY)
    root = _add_node(skill, "0", QUERY, tr.NodeStatus.EXPANDED, depth=0, parent_id=None)
    root.children = ["0.0", "0.1", "0.2"]
    _add_node(skill, "0.0", ANSWERED_Q, tr.NodeStatus.ANSWERED, answer=ANSWER_TEXT)
    _add_node(skill, "0.1", PENDING_Q1, tr.NodeStatus.PENDING)
    _add_node(skill, "0.2", PENDING_Q2, tr.NodeStatus.PENDING)

    result = await skill.assemble_report(QUERY)
    report = result["report_md"]

    assert "## Unresearched Questions" in report, (
        "2 of 4 nodes never researched; the report must say so explicitly, not "
        "just read as complete"
    )
    assert PENDING_Q1 in report and PENDING_Q2 in report, (
        "a reader must see WHICH questions were never answered, not only a count"
    )
    assert "2 of 4" in report, (
        "the coverage fraction (nodes_researched of nodes_total) must be "
        f"unmistakable to a reader. got:\n{report}"
    )
    idx = report.index("## Unresearched Questions")
    assert "not findings" in report[idx:] or "OPEN QUESTIONS" in report[idx:], (
        "the section must label itself as unanswered questions, not findings — "
        "a golden fact merely NAMED in a question must not read as researched"
    )
    assert PENDING_Q1 not in report[:idx] and PENDING_Q2 not in report[:idx], (
        "a PENDING question must not leak into the scored body above this "
        "section — that is the existing rollup()/synthesize_node fail-closed "
        "decision this change must not undo"
    )


# --------------------------------------------------------------------------
# a run with none does not emit a spurious empty section
# --------------------------------------------------------------------------

async def test_report_with_no_pending_nodes_has_no_disclosure_section():
    skill = _bare_skill(QUERY)
    _add_node(skill, "0", QUERY, tr.NodeStatus.ANSWERED, answer=ANSWER_TEXT,
             depth=0, parent_id=None)

    result = await skill.assemble_report(QUERY)

    assert "## Unresearched Questions" not in result["report_md"], (
        "a fully-researched tree must not carry a heading with nothing under it"
    )


# --------------------------------------------------------------------------
# run()'s stats gain a mechanically-readable N/M pair
# --------------------------------------------------------------------------

async def test_run_stats_carry_nodes_researched_and_nodes_total(monkeypatch):
    skill = _bare_skill(QUERY)

    async def fake_research(node: tr.ResearchNode) -> None:
        node.status = tr.NodeStatus.ANSWERED
        node.answer_md = f"Answer for: {node.question}"
        node.answer_digest = node.answer_md
        node.tokens_spent = 10
        node.credits_spent = 0.0

    async def fake_expand(node: tr.ResearchNode) -> list:
        # only the root expands — keeps the tree small and its size predictable
        return [] if node.parent_id is not None else \
            ["Child question A", "Child question B", "Child question C"]

    counter = iter(range(10))

    async def fake_embed(_text: str) -> list:
        vec = [0.0] * 10
        vec[next(counter)] = 1.0  # orthogonal per call: no novelty pruning, no dedup
        return vec

    async def dead_chat(**_kwargs) -> str:
        raise RuntimeError("no llm in this test")  # merge stage fails closed, doesn't crash

    skill.research_node = fake_research
    skill.generate_child_questions = fake_expand
    skill.embed_question = fake_embed
    monkeypatch.setattr(tr, "create_chat_completion", dead_chat)

    result = await skill.run(query=QUERY, max_depth=1, max_breadth=3, max_nodes=2,
                             node_concurrency=3)
    stats = result["stats"]

    assert stats["nodes_total"] == 4, (
        f"root + 3 generated children == 4 nodes in the tree; got stats={stats}"
    )
    assert stats["nodes_researched"] == 2, (
        "max_nodes=2 caps research at the root plus one child, leaving the "
        f"other 2 children PENDING; got stats={stats}"
    )
    assert stats["nodes_researched"] == stats["nodes_total"] - stats["pending_count"], (
        "the new pair must agree with the existing pending_count mechanically, "
        "not carry a second, independently-drifting definition of the same fact"
    )
    assert "## Unresearched Questions" in result["report_md"]
