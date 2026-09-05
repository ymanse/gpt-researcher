"""s11: routing a tree costs ONE query classification, not one per sub-query per node.

Measured 2026-09-05 (harness-search/spec/p0-p1-slimming.md): a single tree node spends
8-9 `claude` CLI sessions and 4-5 of them are the SAME routing decision taken over and
over. `ResearchConductor` runs one search per sub-query, `get_search_results` builds a
fresh `SmartRetriever` for each of those searches (the retriever is stateless by
design -- see the `_DEAD_RETRIEVERS` comment in smart_retriever.py), and every one of
them asks the FAST_LLM "which category is this query". A 20-node tree therefore buys
~80 classifications of questions that all belong to one research run, against a
100-session cap -- the tree is cut short by its own routing overhead.

The category is a property of the RUN, not of the sub-query: `TreeResearchSkill`
resolves it once and hands it to every node researcher
(`researcher.cfg.smart_retriever_force_category`), which `_classify_query` already
honours without an LLM call.

What the fake node researcher stands in for: `conduct_research` here does exactly what
`ResearchConductor._get_context_by_web_search` does to the classifier -- one
`SmartRetriever(sub_query, researcher=self)._classify_query()` per sub-query -- and
nothing else. The classifier that runs is the REAL one, reading the REAL cfg the tree
handed it, so the saving is composed rather than stipulated.

Attribution is by LLM tier: `_classify_query` asks `cfg.fast_llm_model`, every
tree-level call (answer, expansion, merge) asks `cfg.strategic_llm_model`, so one fake
standing in for both `create_chat_completion` bindings can tell a classification from
everything else no matter which binding the resolution ends up going through.

Deterministic: no network, no live LLM, no embeddings service, no container.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import gpt_researcher.skills.tree_research as tr
import gpt_researcher.utils.llm as llm_module
from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE, SmartRetriever

FAST_MODEL = "fast-classifier-sentinel"
STRATEGIC_MODEL = "strategic-writer-sentinel"
# a real Config carries three tiers; choose_agent asks the smart one
SMART_MODEL = "smart-agent-sentinel"

# The classifier answers a DIFFERENT (still valid) category each time it is asked, which
# is what a per-sub-query classifier really behaves like: the sub-queries of one node
# are rephrasings and the FAST_LLM has no reason to land on the same bucket for all of
# them. A run that classifies once therefore routes every node identically; a run that
# classifies per sub-query does not.
VERDICTS = ["academic", "news_current", "code_technical"]
FIRST_VERDICT = VERDICTS[0]
assert all(v in ROUTING_TABLE for v in VERDICTS)

ANSWER_BLOB = "ANSWER: outbox answer body\nDIGEST: outbox digest\nLEARNINGS:\n- one claim\n"

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
CHILD_QS = ["How is outbox table growth bounded in production?",
            "Which brokers deduplicate replayed outbox messages?"]

# One search per sub-query, one SmartRetriever per search (researcher.py
# _get_context_by_web_search -> _process_sub_query -> get_search_results).
SUB_QUERIES = ["outbox failure modes", "outbox replay duplicates"]

DOC_URL = "https://example.test/outbox"
# research_node fails a node closed below MIN_CONTEXT_CHARS, and a FAILED root is never
# expanded, so the stand-in context has to be a plausible size.
CONTEXT = ("The transactional outbox writes the message and the row in one commit. "
           * (tr.MIN_CONTEXT_CHARS // 40))

EMB = {
    ROOT_Q: [1.0, 0.0, 0.0, 0.0],
    CHILD_QS[0]: [0.0, 1.0, 0.0, 0.0],
    CHILD_QS[1]: [0.0, 0.0, 1.0, 0.0],
}


class _LLMBoundary:
    """Stands in for both `create_chat_completion` bindings, attributing by model tier.

    `classify_calls` is the P0 `classify` site: every call that asked the FAST_LLM.
    Appends are made from the classifier's worker thread (`_run_coro_blocking` hops off
    the running loop), but the hop blocks that loop, so the calls stay serialised.
    """

    def __init__(self):
        self.classify_calls = []
        self.other_calls = []

    async def __call__(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None:
            model = next((a for a in args
                          if isinstance(a, str)
                          and a in (FAST_MODEL, STRATEGIC_MODEL, SMART_MODEL)), None)
        if model == FAST_MODEL:
            self.classify_calls.append(kwargs.get("messages"))
            return VERDICTS[(len(self.classify_calls) - 1) % len(VERDICTS)]
        self.other_calls.append(model)
        return ANSWER_BLOB


def _cfg(**over):
    """The config shape a Config instance presents to the classifier.

    `smart_retriever_force_category` is present and None -- that is what
    config/variables/default.py ships -- so a stamped category has to be written, not
    merely defaulted into existence.
    """
    base = dict(fast_llm_provider="fake", fast_llm_model=FAST_MODEL,
                strategic_llm_provider="fake", strategic_llm_model=STRATEGIC_MODEL,
                smart_llm_provider="fake", smart_llm_model=SMART_MODEL,
                llm_kwargs={}, smart_retriever_config=None,
                smart_retriever_force_category=None, config_path=None)
    base.update(over)
    return SimpleNamespace(**base)


def _skill() -> tr.TreeResearchSkill:
    parent = SimpleNamespace(query=ROOT_Q, cfg=_cfg(), tone=None, websocket=None,
                             headers={}, visited_urls=set())
    return tr.TreeResearchSkill(parent)


def _node_researcher_class(built: list):
    class _FakeNodeResearcher:
        """One tree node's GPTResearcher, reduced to its effect on the classifier."""

        def __init__(self, query=None, **kwargs):
            # **kwargs: research_node already passes 8 keywords and P1.2/P1.3 add more
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            self.routed_categories = []
            built.append(self)

        async def conduct_research(self):
            for sub_query in SUB_QUERIES:
                self.routed_categories.append(
                    SmartRetriever(sub_query, researcher=self)._classify_query())
            return CONTEXT

        def get_research_sources(self):
            return [{"url": DOC_URL, "raw_content": CONTEXT}]

        def get_costs(self):
            return 0.0

    return _FakeNodeResearcher


async def _run_tree(max_nodes: int = 3):
    """One hermetic tree run; returns (built node researchers, LLM boundary, result)."""
    skill = _skill()
    built: list = []
    boundary = _LLMBoundary()

    async def _embed(text):
        return list(EMB.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def _children(node):
        return list(CHILD_QS) if node.depth == 0 else []

    skill.embed_question = _embed
    with mock.patch.object(tr, "GPTResearcher", _node_researcher_class(built)), \
         mock.patch.object(tr, "create_chat_completion", new=boundary), \
         mock.patch.object(llm_module, "create_chat_completion", new=boundary), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children):
        result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2,
                                 max_nodes=max_nodes, node_concurrency=3)
    return built, boundary, result


# --------------------------------------------------------------------------- run level

async def test_a_tree_run_pays_for_classification_once_not_once_per_sub_query_per_node():
    """The P1.1 metric itself: `classify` calls per run, cap 2 (spec gate table)."""
    built, boundary, _ = await _run_tree(max_nodes=3)

    assert len(built) == 3, (
        f"the fixture must research three nodes for the per-node cost to be visible, "
        f"got {len(built)} node researchers")
    # EXACTLY one, not "at most one". `<= 1` admits ZERO, and zero is a different
    # implementation the spec explicitly rejects: `self._category = "academic"` hardcoded
    # (or any keyword rule) would satisfy an at-most bound while silently changing which
    # retrievers every node routes to. Routing quality is the whole point of the category,
    # and one call per run fits inside every tier budget, so the run must actually ask.
    assert len(boundary.classify_calls) == 1, (
        f"{len(boundary.classify_calls)} FAST_LLM classifications for one run of "
        f"{len(built)} nodes x {len(SUB_QUERIES)} sub-queries -- expected exactly 1. "
        "More than one means the routing category is re-derived per sub-query per node, "
        "each a `claude` CLI session charged against the run's cap; ZERO means the "
        "category was not resolved by the classifier at all, which changes every node's "
        "retriever bundle without measuring anything")


async def test_every_node_researcher_is_handed_the_category_the_run_resolved():
    """The mechanism: the category reaches the node's cfg, where `_classify_query`
    already short-circuits on it."""
    built, boundary, _ = await _run_tree(max_nodes=3)

    stamped = [r.cfg.smart_retriever_force_category for r in built]
    assert stamped == [FIRST_VERDICT] * len(built), (
        f"node researchers carry {stamped}, expected every one to carry the category the "
        f"run resolved ({FIRST_VERDICT!r}); an unstamped cfg means each node re-classifies "
        "every sub-query from scratch")


async def test_all_nodes_of_one_run_route_to_the_same_category():
    """A category is a property of the run, so no two nodes may search different
    retriever bundles for the same research question."""
    built, _, _ = await _run_tree(max_nodes=3)

    routed = [c for r in built for c in r.routed_categories]
    assert len(routed) == len(built) * len(SUB_QUERIES), "every sub-query must be routed"
    assert set(routed) == {FIRST_VERDICT}, (
        f"one run routed its sub-queries to {sorted(set(routed))} -- each sub-query "
        "classified itself, so sibling searches of one tree hit different retriever "
        f"bundles instead of the run's single category {FIRST_VERDICT!r}")


async def test_the_run_still_answers_every_node_and_writes_a_report():
    """The saving must not be bought by degrading the run: nodes still answer and the
    roll-up still runs."""
    built, _, result = await _run_tree(max_nodes=3)

    assert result["stats"]["researched"] == 3, (
        f"nodes researched dropped to {result['stats']['researched']} of 3 -- the "
        "classification saving must not cost the tree any research")
    assert result["report_md"].strip(), "the run produced no report body"


# ------------------------------------------------------- classifier contract (guards)

def test_a_forced_category_is_honoured_without_asking_the_classifier_llm():
    """Regression guard -- this short-circuit is what makes the stamping worth anything."""
    boundary = _LLMBoundary()
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("anything", cfg=_cfg(
            smart_retriever_force_category="academic"))._classify_query()

    assert category == "academic", f"the forced category must be used verbatim, got {category!r}"
    assert boundary.classify_calls == [], (
        f"{len(boundary.classify_calls)} FAST_LLM calls were made although the category "
        "was already decided -- the forced category has to cost nothing, or handing it to "
        "every node saves nothing")


def test_an_unset_force_category_still_asks_the_classifier_llm():
    """Regression guard: today's behaviour is unchanged when nothing forces a category."""
    boundary = _LLMBoundary()
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("anything", cfg=_cfg())._classify_query()

    assert len(boundary.classify_calls) == 1, (
        f"{len(boundary.classify_calls)} FAST_LLM calls for an unforced query -- an unset "
        "category must still be classified exactly once, not skipped and not repeated")
    assert category == FIRST_VERDICT, (
        f"the classifier's own verdict must be used, got {category!r}")


def test_a_force_category_outside_the_routing_table_falls_back_to_the_classifier():
    """A typo in SMART_RETRIEVER_FORCE_CATEGORY must not route to a category that has no
    retriever bundle -- `_route_to_retrievers` would silently serve general_web instead."""
    boundary = _LLMBoundary()
    bogus = "acedemic"
    assert bogus not in ROUTING_TABLE
    with mock.patch.object(llm_module, "create_chat_completion", new=boundary):
        category = SmartRetriever("anything", cfg=_cfg(
            smart_retriever_force_category=bogus))._classify_query()

    assert category != bogus, (
        f"an unknown forced category {bogus!r} was routed on as-is; it has no entry in "
        "ROUTING_TABLE, so the search would fall through to a bundle nobody chose")
    assert category in ROUTING_TABLE and len(boundary.classify_calls) == 1, (
        f"an unknown forced category must fall through to the classifier: got "
        f"{category!r} after {len(boundary.classify_calls)} FAST_LLM calls")
