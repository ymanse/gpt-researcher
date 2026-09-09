"""s11 / P1.3: a node question too long to work as a search query must be planned.

P1.3 hands every node `preset_sub_queries=[node.question]` so the node skips the planner's
probe search and its planning LLM call. The justification is that "the node's question IS
its plan" - true of a CHILD question, which an LLM wrote to be searchable. It is not true
of the ROOT question, which is whatever the user typed.

Measured 2026-09-09, the run this pins. The root below is that run's real question,
verbatim from `D:/dev_ext/gptr-mcp/outputs/sk-ax-...-21cbb838.tree.json`: 390 characters
of Korean naming a company, a jobs site, an aptitude test and an AI platform strategy in
four numbered parts. The preset handed all 390 characters to the retrievers as ONE search
string. serper returned 2 results and firecrawl 3; the node fell below MIN_CONTEXT_CHARS
and failed closed. The tree ended at node_count 1, status failed, 0 citations, having
spent 3 CLI sessions on nothing. Before P1.3 the planner would have decomposed it.

So the preset is withheld above a length threshold, and the check runs on ANY node rather
than only the root: children are shorter than this root, and it self-protects if a child
ever comes back long.

Hermetic: the node researcher, the answer LLM and the child-question generator are all
replaced, so no network, no LLM, no embeddings client. Nothing here arms the CLI budget.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.tree_research as tr


# Verbatim from the failed 2026-09-09 run rather than a synthetic string of round length:
# the claim being pinned is that a question a real user actually typed crosses the
# threshold, and its exact width is the evidence for where the threshold sits.
LONG_ROOT = "SK AX (SK inc. C&C \uc0ac\uc5c5\ubd80, \uad6c SK C&C, \ubd84\ub2f9 \ud310\uad50) \uacbd\ub825\uc9c1 \ucc44\uc6a9 \uc9c0\uc6d0 \uc900\ube44 2025~2026: (1) SK Careers(skcareers.com) \uacbd\ub825 \uc9c0\uc6d0\uc11c \uc591\uc2dd\u00b7\uc790\uae30\uc18c\uac1c\uc11c \ubb38\ud56d\u00b7\uae00\uc790\uc218\uc640 \uc11c\ub958 \ud3c9\uac00 \ud3ec\uc778\ud2b8, (2) SKCT \uc2ec\uce35\uc5ed\ub7c9\uac80\uc0ac \uacbd\ub825\uc9c1 \uc720\ud615\u00b7\uad6c\uc131\u00b7\uc900\ube44\ubc95, (3) 1\ucc28\u00b72\ucc28 \uba74\uc811 \ubc29\uc2dd(\uae30\uc220\u00b7\uc5ed\ub7c9)\uacfc \ucc44\uc6a9\uac80\uc9c4/\ucc98\uc6b0\ud611\uc758 \uad00\ud589\u00b7\uc5f0\ubd09 \uc218\uc900(\uc804\ubb38\uac00\u00b710\ub144+ \uc544\ud0a4\ud14d\ud2b8), (4) SK AX \uc758 2025~2026 AI \uc0ac\uc5c5 \ubc29\ud5a5 \u2014 AI Biz. Innovation / AI Tech Innovation \uc870\uc9c1, Agentic AI \ud50c\ub7ab\ud3fc, A2A/MCP \uae30\ubc18 Agent Orchestration Platform, \uc628\ud504\ub808\ubbf8\uc2a4\u00b7\ud3d0\uc1c4\ub9dd \uace0\uac1d\uc0ac \ub51c\ub9ac\ubc84\ub9ac \uc0ac\ub840, \uc81c\uc870\u00b7\uae08\uc735 AX \ud504\ub85c\uc81d\ud2b8"

# Short, English, LLM-shaped - the kind of question `generate_child_questions` produces.
SHORT_CHILDREN = [
    "SK Careers career application essay questions and character limits",
    "SKCT aptitude test format for experienced hires",
]

ANSWER = "ANSWER: answer body\nDIGEST: digest\nLEARNINGS:\n- one claim\n"
DOC_URL = "https://example.test/skax"

# `research_node` multiplies this cap by `max_iterations + 1` to compensate for the preset
# researching one sub-query where the planner researched four. Both are default.py values.
BASE_CAP = 5
MAX_ITERATIONS = 3


def _cfg():
    return SimpleNamespace(
        fast_llm_provider="fake", fast_llm_model="fast-sentinel",
        strategic_llm_provider="fake", strategic_llm_model="strategic-sentinel",
        smart_llm_provider="fake", smart_llm_model="smart-sentinel",
        llm_kwargs={}, smart_retriever_config=None,
        smart_retriever_force_category=None, config_path=None,
        max_search_results_per_query=BASE_CAP, max_iterations=MAX_ITERATIONS)


async def _run_tree() -> dict:
    """One tree run over the real long root plus two short children, keyed by question.

    Both halves of the rule are read off the same run on purpose: an implementation that
    switched the preset off everywhere would satisfy the long-question test alone while
    handing back the two sessions per node that P1.3 bought.
    """
    # research_node fails a node closed below MIN_CONTEXT_CHARS and a FAILED root is never
    # expanded, so the stand-in context has to be a plausible size or no child is built.
    context = "SK AX runs an Agentic AI platform delivery organisation. " * (
        tr.MIN_CONTEXT_CHARS // 40)
    built: list = []

    class _FakeNodeResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            built.append(self)

        async def conduct_research(self):
            return context

        def get_research_sources(self):
            return [{"url": DOC_URL, "raw_content": context}]

        def get_costs(self):
            return 0.0

    # Orthogonal vectors: a child at cosine >= DEDUP_COSINE against a question already in
    # the tree is dropped, which would silently shrink the node count this run asserts on.
    vectors = {LONG_ROOT: [1.0, 0.0, 0.0, 0.0],
               SHORT_CHILDREN[0]: [0.0, 1.0, 0.0, 0.0],
               SHORT_CHILDREN[1]: [0.0, 0.0, 1.0, 0.0]}

    async def _embed(text):
        return list(vectors.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def _children(node):
        return list(SHORT_CHILDREN) if node.depth == 0 else []

    parent = SimpleNamespace(query=LONG_ROOT, cfg=_cfg(), tone=None,
                             websocket=None, headers={}, visited_urls=set())
    skill = tr.TreeResearchSkill(parent)
    skill.embed_question = _embed

    with mock.patch.object(tr, "GPTResearcher", _FakeNodeResearcher), \
         mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value=ANSWER)), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children):
        await skill.run(query=LONG_ROOT, max_depth=1, max_breadth=2,
                        max_nodes=3, node_concurrency=3)

    assert len(LONG_ROOT) == 390, (
        "fixture check: this must stay the verbatim 390-character root of the failed "
        f"2026-09-09 run, measured {len(LONG_ROOT)} characters")
    by_question = {r.query: r for r in built if r.query in [LONG_ROOT] + SHORT_CHILDREN}
    assert len(by_question) == 3, (
        "fixture precondition: one researcher per node for three nodes, measured "
        f"{len(by_question)} - a root that failed closed is never expanded, so a short "
        "count here means the stand-in context landed under MIN_CONTEXT_CHARS")
    return by_question


@pytest.mark.asyncio
async def test_a_question_too_long_to_be_a_search_query_is_planned_instead_of_presetted():
    """The measured failure: 390 characters handed to the retrievers as one search string.

    serper answered it with 2 results and firecrawl with 3, the node fell below
    MIN_CONTEXT_CHARS and failed closed, and the tree finished at node_count 1 / status
    failed / 0 citations having spent 3 CLI sessions. Withholding the preset costs the
    planner's two sessions for that node, which is cheaper than a node that researches
    nothing.
    """
    nodes = await _run_tree()

    root_preset = nodes[LONG_ROOT].ctor_kwargs.get("preset_sub_queries")
    assert not root_preset, (
        f"the {len(LONG_ROOT)}-character root question was handed to its own researcher "
        f"as its sub-query list, so the retrievers search that entire string verbatim: "
        f"measured preset {root_preset!r}. That is the 2026-09-09 run - serper 2 results, "
        f"firecrawl 3, context under MIN_CONTEXT_CHARS ({tr.MIN_CONTEXT_CHARS}), node "
        f"failed, tree finished at node_count 1 with 0 citations")


@pytest.mark.asyncio
async def test_a_short_question_in_the_same_run_still_bypasses_the_planner():
    """The rule must cost nothing on the questions P1.3 was written for."""
    nodes = await _run_tree()

    lost = {q: nodes[q].ctor_kwargs.get("preset_sub_queries") for q in SHORT_CHILDREN
            if nodes[q].ctor_kwargs.get("preset_sub_queries") != [q]}
    assert not lost, (
        f"{len(lost)} of {len(SHORT_CHILDREN)} short child questions "
        f"({[len(q) for q in SHORT_CHILDREN]} characters) lost their preset, so each pays "
        f"the planner's probe search plus its planning LLM call again - the 2 of 8 "
        f"sessions per node P1.3 exists to remove. measured presets: {lost}")


@pytest.mark.asyncio
async def test_withholding_the_preset_also_withholds_the_widened_result_cap():
    """The cap widening is the preset's other half and must not outlive it.

    `research_node` multiplies `max_search_results_per_query` by `max_iterations + 1`
    precisely because the preset researches ONE sub-query where the planner researched
    four. Leave it on while the planner runs and each of those four sub-queries is allowed
    four times the documents - 16x the cap, four times what the pre-P1.3 planner path ever
    read, in scraping and context nobody asked for.
    """
    nodes = await _run_tree()

    root_cap = nodes[LONG_ROOT].cfg.max_search_results_per_query
    assert root_cap == BASE_CAP, (
        f"the planner-path root kept the preset's widened cap {root_cap}, so its "
        f"{MAX_ITERATIONS + 1} planned sub-queries may read up to "
        f"{root_cap * (MAX_ITERATIONS + 1)} documents against the "
        f"{BASE_CAP * (MAX_ITERATIONS + 1)} the planner path read before P1.3")

    widened = BASE_CAP * (MAX_ITERATIONS + 1)
    thinned = [q for q in SHORT_CHILDREN
               if nodes[q].cfg.max_search_results_per_query != widened]
    assert not thinned, (
        f"a presetted node lost the compensating cap: measured "
        f"{[nodes[q].cfg.max_search_results_per_query for q in thinned]} against "
        f"{widened}, which is the 4x context collapse (45-60k chars down to 13-15k) that "
        f"widening the cap fixed")
