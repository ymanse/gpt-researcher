"""s5 round 2 — the two gaps bench round 3 and the s5 review left open.

(1) S6 = contested-병기 rate × (1 - contradiction/unsupported penalty). Round 3
    measured the penalty at 0 on all five goldens and S6 still came in at 33
    against the baseline's 52 — the whole shortfall is the 병기 half: the report
    states one side of a controversy and never the other. Of the 11 contested
    sides the goldens ask for and the reports miss, 8 are missing from EVERY node
    answer as well. Since synthesize_node only CONCATENATES node answers, the
    answer prompt is the last place a disagreement can survive: whatever it
    collapses is gone from the report for good.

(2) Review F1 (minor): the separator citation-marker strip added last round runs
    on the RAW slice, and _scrubbed() blanks a fenced code block exactly like a
    marker. A fence abutting a dropped sentence's period lands inside that
    separator, so a bracket in the code ("data[1] = load()") got deleted out of
    the shipped report — silent content corruption, not a metric leak. Two of the
    five goldens are code-heavy, one of them a measure_pair query.

Deterministic, no network, no LLM.
"""
import re
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod

CONTEXT_MARK = "Collected page text from the scraped sources."
RICH_CONTEXT = (CONTEXT_MARK + " ") * 400  # ~18k chars, over MIN_CONTEXT_CHARS
NODE_Q = "How large was the ported codebase?"
NODE_A = "One report puts the port at 530,000 lines; another counts 960,000."


def _bare_skill():
    parent = SimpleNamespace(
        query=NODE_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    return tree_mod.TreeResearchSkill(parent)


# ---------------------------------------------------------------------------
# (1) the ANSWER prompt has to demand every side of a disagreement
# ---------------------------------------------------------------------------

async def test_answer_prompt_demands_every_side_of_a_disagreement(monkeypatch):
    prompts = []

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            return RICH_CONTEXT

        def get_research_sources(self):
            return [{"url": "https://example.com/doc",
                     "title": "doc", "raw_content": (NODE_A + " Supporting prose. ") * 8}]

        def get_costs(self):
            return 0.0

    async def fake_chat(messages=None, **kwargs):
        prompts.append(" ".join(str(m.get("content", "")) for m in (messages or [])))
        return f"ANSWER: {NODE_A}\nDIGEST: {NODE_A}\nLEARNINGS:\n- {NODE_A}\n"

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    skill = _bare_skill()
    node = tree_mod.ResearchNode(id="0", question=NODE_Q, parent_id=None, depth=0)
    await skill.research_node(node)

    assert prompts, "research_node must ask the answer LLM at all"
    prompt = prompts[0].lower()

    assert "disagree" in prompt, (
        "the answer prompt never mentions disagreement, so a node that read two "
        "sources reporting different figures for the same quantity is free to "
        "pick one. The roll-up only concatenates node answers — the side dropped "
        "here never reaches the report, and S6's contested-병기 half is what "
        "measures that loss (bench round 3: S6 33 vs baseline 52, penalty 0). "
        f"got: {prompts[0]!r}"
    )
    assert re.search(r"every side|both sides|all sides", prompt), (
        "naming disagreement is not enough — the prompt has to say what to DO "
        "with it: state every side explicitly"
    )
    assert re.search(r"never (average|drop)|not average", prompt), (
        "'53,000 to 530,000 lines' or a midpoint is not 병기: an averaged or "
        "range-merged figure matches neither contested value and can land "
        "squarely on a trap pattern (bun-rust-port traps a wrong-magnitude line "
        "count). The prompt must forbid merging and forbid dropping the "
        "minority view."
    )

    words = re.search(r"<=\s*(\d+)\s*words", prompt)
    assert words and int(words.group(1)) >= 600, (
        "both sides of every contested point do not fit in the old 400-word "
        "budget: the node compresses up to 60,000 chars of context into this "
        f"answer. got cap={words.group(1) if words else None!r}"
    )


# ---------------------------------------------------------------------------
# (2) review F1: dropping a sentence must not rewrite the code fence after it
# ---------------------------------------------------------------------------

FENCE_BODY = (
    "Quarterly kumquat pallet shipments reached 48,500 units."
    "```data[1] = load()```"
    "\nThe harness sharded the work across git worktrees.\n"
)


async def test_dropping_a_claim_leaves_the_code_fence_after_it_intact():
    """No node answer covers 48,500, so the first sentence is dropped as
    unsupported and the strip runs over the separator that follows it — which,
    in the scrubbed view, swallows the whole fence."""
    skill = _bare_skill()
    node = tree_mod.ResearchNode(id="0", question=NODE_Q, parent_id=None, depth=0)
    node.status = tree_mod.NodeStatus.ANSWERED
    node.answer_md = NODE_A
    skill.nodes["0"] = node

    clean, contradictions, unsupported = skill.verify_rollup(FENCE_BODY)

    assert any("48,500" in u for u in unsupported), (
        "fixture guard: the claim has to be dropped for the strip to run at all. "
        f"got unsupported={unsupported!r} contradictions={contradictions!r}"
    )
    assert "data[1] = load()" in clean, (
        "the marker strip may only remove the dropped sentence's orphaned "
        "citation markers. A bracket inside a fenced block is CODE — deleting it "
        "ships a corrupted example to the reader, and nothing downstream ever "
        f"notices because the scorer scrubs fences before it reads. got: {clean!r}"
    )


async def test_dropping_a_claim_still_removes_its_orphaned_citation_marker():
    """No-regression for the fix above: outside a fence the strip must still fire,
    or a dropped claim leaves an [id] sitting on text it never supported."""
    skill = _bare_skill()
    node = tree_mod.ResearchNode(id="0", question=NODE_Q, parent_id=None, depth=0)
    node.status = tree_mod.NodeStatus.ANSWERED
    node.answer_md = NODE_A
    skill.nodes["0"] = node

    body = ("Quarterly kumquat pallet shipments reached 48,500 units. [7]\n"
            "The harness sharded the work across git worktrees.\n")
    clean, _, unsupported = skill.verify_rollup(body)

    assert any("48,500" in u for u in unsupported), "fixture guard: claim must drop"
    assert "[7]" not in clean, (
        "the dropped sentence's trailing marker now points at the sentence after "
        f"it — an ungrounded [id] the S1 scorer counts against us. got: {clean!r}"
    )
