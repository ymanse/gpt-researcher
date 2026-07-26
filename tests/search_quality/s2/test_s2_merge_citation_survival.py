"""Regression guard — citations must survive a multi-child node roll-up.

Root cause (measured live across three progressively wider fixes: match
against raw source pages, then immediate pre-merge blocks, then the whole
subtree's pre-merge blocks): asking an LLM to rewrite several already-cited
passages into one merged passage, then re-deriving [id] markers by fuzzy
text-matching the rewrite against pre-merge text, loses more citations at
every rewrite level -- culminating in citations_total=0 for a 33-node,
depth-3 tree (bun-rust-port) even with the widest match corpus. synthesize_node
no longer rewrites at all for a multi-child node: it returns the node's own
attributed text followed by each child's summary verbatim, so a child's [id]
markers reach the parent (and the root) untouched. Deterministic, no network.
"""
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod


def _skill() -> tree_mod.TreeResearchSkill:
    return tree_mod.TreeResearchSkill(SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    ))


def _node(question: str = "root question") -> tree_mod.ResearchNode:
    return tree_mod.ResearchNode(id="0", question=question, parent_id=None,
                                 depth=0, status=tree_mod.NodeStatus.ANSWERED)


async def test_own_and_child_citations_both_survive_the_rollup():
    skill = _skill()
    node = _node()
    node.answer_md = "The team ported the codebase to Rust in eleven days in 2024."
    node.sources = ["https://a.example.com"]
    skill._read_docs["https://a.example.com"] = (
        "filler " * 50 + "the team ported the codebase to rust in eleven days "
        "in 2024 " + "filler " * 50
    )
    child_summary = "The outbox pattern avoided dual-write inconsistency [7]."

    result = await skill.synthesize_node(
        node, child_summaries=[child_summary],
        source_ids={"https://a.example.com": "1"})

    assert "[1]" in result, "own finding's citation must survive the rollup"
    assert "[7]" in result, "child summary's citation must survive the rollup"
    assert child_summary in result, "child summary text must pass through verbatim"


async def test_no_llm_call_for_a_multi_child_rollup(monkeypatch):
    """A rewrite step is what erased citations in every prior design -- guard
    against reintroducing one for the multi-child path."""
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("synthesize_node must not call the LLM to merge children")

    monkeypatch.setattr(tree_mod, "create_chat_completion", fail_if_called)
    skill = _skill()
    node = _node()
    node.answer_md = "Own finding."
    node.sources = []

    result = await skill.synthesize_node(
        node, child_summaries=["A child finding [3]."], source_ids={})

    assert "[3]" in result


async def test_grandchild_citation_survives_three_levels_of_rollup():
    """The regression this whole suite guards: citations must not compound-lose
    going up a multi-level tree (measured live: citations_total=0 by the root
    of a depth-3 tree under every rewrite-based design)."""
    skill = _skill()
    grandchild_summary = "The org migrated billing infrastructure to a new stack [4]."
    child_summary = f"Overall the org modernized several core systems. {grandchild_summary}"

    root = _node("root question")
    root.answer_md = "An unrelated root-level finding entirely."
    root.sources = []

    result = await skill.synthesize_node(
        root, child_summaries=[child_summary], source_ids={})

    assert "[4]" in result, (
        "a citation earned deep in the subtree must still be present at the root "
        "once every level simply concatenates rather than rewrites"
    )


async def test_uncited_own_text_gets_no_fabricated_marker():
    skill = _skill()
    node = _node()
    node.answer_md = "Unrelated finding about a different topic entirely."
    node.sources = ["https://a.example.com"]
    skill._read_docs["https://a.example.com"] = "completely different unrelated page content"

    result = await skill.synthesize_node(
        node, child_summaries=["Another child sub-finding [9]."],
        source_ids={"https://a.example.com": "1"})

    assert "[1]" not in result, (
        "own text with no wording overlap against its source must not be cited"
    )
    assert "[9]" in result, "the child's own citation is untouched by the parent's own text"
