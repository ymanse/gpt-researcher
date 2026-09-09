"""s11: the LINEAR deep-research path pays its run-level costs ONCE, and degrades.

Measured on the MCP container 2026-09-09. A tree run and a linear run were fired four
minutes apart. Allowances POOL (see `begin_agent_run`), so the two shared one ceiling
of 53; the linear run burned all of it and died with `AgentBudgetExceeded`, 22 rounds
of it, and the whole call surfaced as "Failed to get response from claude_agent API" —
nothing of the queries that had already finished came back.

Two defects, both in `DeepResearchSkill`, both already solved on the tree side:

  A. Every nested `GPTResearcher` is built with no agent, no role and no forced routing
     category, so each sub-query pays `choose_agent` once and a FAST_LLM classification
     once. The classification half is a REGRESSION: until 2026-09-06 the retriever was
     constructed without the researcher, `SmartRetriever.cfg` was None, and
     `_classify_query` answered "general_web" on its very first branch for ZERO
     sessions. Passing the researcher fixed smart routing — a real bug — and switched
     those sessions on. The tree caps them at one per run by forcing the category
     (`TreeResearchSkill._resolve_run_context`); the linear path has no such cap.

  B. `deep_research` never asks whether the budget can pay for the work it is about to
     start. The tree tests `agent_budget_exhausted(reserve=agent_synthesis_reserve())`
     between node batches and finishes with a shallower report. Linear runs until a
     charge raises — and the recursion's `generate_search_queries` sits outside every
     `except`, so that raise takes the entire research call with it.

WHAT IS COUNTED. Sessions spawned, the unit the budget bills in, attributed by which
`create_chat_completion` binding they went through — the three are disjoint because each
is resolved somewhere else: `actions.agent_creator`'s (choose_agent), `utils.llm`'s (the
classifier imports it at CALL time), and `skills.deep_research`'s own module-level one
(this file's strategic planning calls).

WHAT IS NOT PINNED: `preset_sub_queries`. The tree hands each node `[node.question]`
because a node IS one question; the linear path must not, and one test below holds that
open — the nested researcher's own planning is the second level of decomposition that
linear depth is made of.

Deterministic: no network, no live LLM, no container. The budget group arms the REAL
process-wide counter and charges it for real, so it disarms on teardown — s11 sorts
before every other stage and a leaked allowance changes what they measure — and stubs
`prune_cli_sessions`, which deletes transcripts under the operator's real ~/.claude.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.deep_research as dr
import gpt_researcher.utils.llm as llm_module
from gpt_researcher.actions import agent_creator
from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.prompts import PromptFamily
from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE, SmartRetriever
from gpt_researcher.skills.researcher import ResearchConductor

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"

CATEGORY = "academic"
assert CATEGORY in ROUTING_TABLE  # or _classify_query falls through and re-asks

AGENT = "Data Engineering Agent"
ROLE = "You are a data engineering researcher. Answer with sourced specifics."
AGENT_JSON = '{"server": "' + AGENT + '", "agent_role_prompt": "' + ROLE + '"}'

DOC_URL = "https://example.invalid/outbox"
CONTEXT = "The outbox table grows without bound unless a cleanup job trims it. " * 40

# One blob serves both parsers this skill owns: generate_search_queries reads only the
# Query:/Goal: lines, process_research_results only the Learning/Question: ones.
LLM_BLOB = (
    "Query: how production teams bound outbox table growth\n"
    "Goal: find retention practice\n"
    "Query: which brokers deduplicate replayed outbox messages\n"
    "Goal: find dedup guarantees\n"
    f"Learning [{DOC_URL}]: The outbox table grows without bound unless a job trims it.\n"
    "Question: What trims the outbox table in practice?\n"
)

# breadth=2, depth=2 -> 2 top-level queries, each recursing into 2 more: 6 nested
# researchers. The whole point is that the per-run costs must NOT scale with this.
BREADTH, DEPTH = 2, 2
FULL_EXPANSION = 6

# One search per sub-query, one SmartRetriever per search, one classification per
# retriever (researcher.py _get_context_by_web_search -> get_search_results).
SUB_QUERIES = ["outbox failure modes", "outbox replay duplicates"]


def _cfg(**over):
    """The config surface these two paths read. `smart_retriever_force_category` is
    present and None — what config/variables/default.py ships — so a stamped category
    has to be written rather than merely defaulted into existence."""
    base = dict(
        fast_llm_provider="fake", fast_llm_model="fast-model",
        strategic_llm_provider="fake", strategic_llm_model="strategic-model",
        smart_llm_provider="fake", smart_llm_model="smart-model",
        llm_kwargs={}, reasoning_effort="low", curate_sources=False,
        smart_retriever_config=None, smart_retriever_force_category=None,
        config_path=None, deep_research_breadth=BREADTH, deep_research_depth=DEPTH,
        deep_research_concurrency=1, max_iterations=1, max_search_results_per_query=5,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _parent(**cfg_over):
    """The researcher DeepResearchSkill hangs off. A namespace and not a real
    GPTResearcher: constructing one reads the environment's .env and retriever list,
    and nothing here needs either."""
    return SimpleNamespace(
        query=ROOT_Q, cfg=_cfg(**cfg_over), tone=None, websocket=None, headers={},
        visited_urls=set(), mcp_configs=None, mcp_strategy=None, research_sources=[],
        parent_query="", prompt_family=PromptFamily, retrievers=[], log_handler=None,
        add_costs=lambda *a, **k: None, get_costs=lambda: 0.0,
    )


class _Boundary:
    """The three `create_chat_completion` bindings, counted apart.

    `charge` makes every fake call spend one CLI session against the real budget, which
    is what a claude_agent-backed run does — one LLM call, one session.
    """

    def __init__(self, charge: bool = False):
        self.agent_calls: list = []
        self.classify_calls: list = []
        self.strategic_calls: list = []
        self._charge = charge

    def _spend(self):
        if self._charge:
            sub.note_agent_call()

    async def agent(self, *args, **kwargs):
        self.agent_calls.append(kwargs.get("messages"))
        self._spend()
        return AGENT_JSON

    async def classify(self, *args, **kwargs):
        self.classify_calls.append(kwargs.get("messages"))
        self._spend()
        return CATEGORY

    async def strategic(self, *args, **kwargs):
        self.strategic_calls.append(kwargs.get("messages"))
        self._spend()
        return LLM_BLOB


# --------------------------------------------------------------------------- group A
# What one nested researcher costs. The stand-in below keeps the REAL guard
# (conduct_research's `if not (agent and role)`) and the REAL classifier short-circuit,
# so the saving is composed out of shipped code rather than stipulated here.


class _FakeRetriever:
    """conduct_research reads `r.__name__` off every configured retriever."""


def _nested_class(built: list):
    class _NestedResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.agent = kwargs.get("agent")
            self.role = kwargs.get("role")
            visited = kwargs.get("visited_urls")
            self.visited_urls = set() if visited is None else visited
            self.headers = kwargs.get("headers") or {}
            self.websocket = kwargs.get("websocket")
            self.report_source = kwargs.get("report_source") or "web"
            self.report_type = kwargs.get("report_type") or "research_report"
            self.verbose = False
            self.parent_query = ""
            self.prompt_family = PromptFamily
            self.retrievers = [_FakeRetriever]
            self.source_urls = None
            self.complement_source_urls = False
            self.query_domains = []
            self.documents = None
            self.vector_store = None
            self.context = []
            self.research_sources = [{"url": DOC_URL, "raw_content": CONTEXT}]
            self.routed: list = []
            self.cfg = _cfg()
            self.research_conductor = ResearchConductor(self)
            built.append(self)

        def add_costs(self, *args, **kwargs):
            pass

        def get_costs(self):
            return 0.0

        async def conduct_research(self):
            return await self.research_conductor.conduct_research()

        def get_research_sources(self):
            return list(self.research_sources)

    return _NestedResearcher


async def _classifying_web_search(self, query, *args, **kwargs):
    """Stands in for `_get_context_by_web_search`, reduced to its effect on the
    classifier: one real SmartRetriever(sub_query, researcher=...)._classify_query()
    per sub-query, reading the cfg the skill handed this nested researcher."""
    for sub_query in SUB_QUERIES:
        self.researcher.routed.append(
            SmartRetriever(sub_query, researcher=self.researcher)._classify_query())
    return [CONTEXT]


async def _run_linear(monkeypatch):
    """One hermetic linear run; returns (skill, nested researchers, boundary, result)."""
    built: list = []
    boundary = _Boundary()
    # A budget another test armed is process-wide, and this path now stops expanding
    # on it — which would decide these tests for a reason that has nothing to do with
    # what they measure. Patched where it LIVES, not where deep_research imports it:
    # that import is function-local, so it resolves this attribute at call time.
    monkeypatch.setattr(sub, "agent_budget_exhausted", lambda reserve=0: False)
    skill = dr.DeepResearchSkill(_parent())
    with mock.patch("gpt_researcher.GPTResearcher", _nested_class(built)), \
         mock.patch.object(dr, "create_chat_completion", new=boundary.strategic), \
         mock.patch.object(llm_module, "create_chat_completion", new=boundary.classify), \
         mock.patch.object(agent_creator, "create_chat_completion", new=boundary.agent), \
         mock.patch.object(ResearchConductor, "_get_context_by_web_search",
                           new=_classifying_web_search):
        result = await skill.deep_research(query=ROOT_Q, breadth=BREADTH, depth=DEPTH)
    return skill, built, boundary, result


@pytest.mark.asyncio
async def test_a_linear_run_chooses_its_agent_once_not_once_per_nested_researcher(
        monkeypatch):
    """`<= 1` and not `== 1`: resolving the pair without an LLM also satisfies the
    spec. What may never happen is the count scaling with the sub-queries."""
    _, built, boundary, _ = await _run_linear(monkeypatch)

    assert len(built) == FULL_EXPANSION, (
        f"fixture precondition: breadth={BREADTH} depth={DEPTH} must build "
        f"{FULL_EXPANSION} nested researchers, got {len(built)}")
    assert len(boundary.agent_calls) <= 1, (
        f"{len(built)} nested researchers cost {len(boundary.agent_calls)} "
        "agent-selection CLI sessions — the selection is re-run per sub-query although "
        "its answer cannot change within one research run, and that count scales with "
        "breadth x depth against a ceiling measured at 53")


@pytest.mark.asyncio
async def test_a_linear_run_classifies_its_routing_category_once_for_the_whole_run(
        monkeypatch):
    """EXACTLY one, not "at most one". Zero would mean the category was hardcoded
    rather than resolved, which silently changes which retrievers every sub-query
    searches — the routing this classifier exists to get right."""
    _, built, boundary, _ = await _run_linear(monkeypatch)

    assert len(built) == FULL_EXPANSION, (
        f"fixture precondition: {FULL_EXPANSION} nested researchers, got {len(built)}")
    assert len(boundary.classify_calls) == 1, (
        f"{len(boundary.classify_calls)} FAST_LLM classifications for one run of "
        f"{len(built)} nested researchers x {len(SUB_QUERIES)} sub-queries — expected "
        "exactly 1. More means the routing category is re-derived per sub-query, each a "
        "CLI session (a NEW cost since 2026-09-06, when the retriever started receiving "
        "the researcher and cfg stopped being None); zero means nobody asked the "
        "classifier at all")


@pytest.mark.asyncio
async def test_every_nested_researcher_is_handed_the_run_agent_role_and_category(
        monkeypatch):
    """The mechanism: (agent, role) through the constructor — the guard is an AND, so
    half a pair buys nothing — and the category onto the cfg, where `_classify_query`
    already short-circuits on it."""
    _, built, _, _ = await _run_linear(monkeypatch)

    unresolved = [r for r in built
                  if not (r.ctor_kwargs.get("agent") and r.ctor_kwargs.get("role"))]
    assert not unresolved, (
        f"{len(unresolved)} of {len(built)} nested researchers were built without a "
        f"resolved (agent, role) pair, so each re-runs the selection itself: "
        f"{[(r.ctor_kwargs.get('agent'), r.ctor_kwargs.get('role')) for r in built]}")
    pairs = {(r.ctor_kwargs.get("agent"), r.ctor_kwargs.get("role")) for r in built}
    assert len(pairs) == 1, (
        f"the sub-queries of one run were given {len(pairs)} different (agent, role) "
        f"pairs ({pairs}); the selection is a property of the run")

    stamped = [r.cfg.smart_retriever_force_category for r in built]
    assert stamped == [CATEGORY] * len(built), (
        f"nested researchers carry {stamped}, expected every one to carry the category "
        f"the run resolved ({CATEGORY!r}); an unstamped cfg means each sub-query "
        "re-classifies from scratch")
    routed = [c for r in built for c in r.routed]
    assert len(routed) == len(built) * len(SUB_QUERIES), "every sub-query must be routed"
    assert set(routed) == {CATEGORY}, (
        f"one run routed its sub-queries to {sorted(set(routed))} — sibling searches of "
        f"one research question hit different retriever bundles instead of the run's "
        f"single category {CATEGORY!r}")


@pytest.mark.asyncio
async def test_the_nested_researchers_still_plan_their_own_sub_queries(monkeypatch):
    """Deliberately NOT the tree's third slimming, and this pins it open.

    A tree node is one question, so `preset_sub_queries=[node.question]` costs it
    nothing. A linear sub-query is not: the nested researcher's own planning is the
    SECOND level of decomposition, and depth is made of exactly that. Presetting it
    would make depth=2 mean something else while every count in this file still
    passed."""
    _, built, _, _ = await _run_linear(monkeypatch)

    presets = [r.ctor_kwargs["preset_sub_queries"] for r in built
               if r.ctor_kwargs.get("preset_sub_queries")]
    assert not presets, (
        f"{len(presets)} nested researchers were built with preset_sub_queries "
        f"({presets[:2]}), which skips the planner that produces the second level of "
        "decomposition — the run would still be called depth=2 while researching one")


# --------------------------------------------------------------------------- group B
# A spent budget must cost the run its DEPTH, not its result.

ALLOWANCE = 8
# agent_synthesis_reserve() = max(5, 15% of the allowance) = 5 here, so exactly 3
# sessions sit above the reserve: resolving the run context (1) and planning the
# top-level queries (1) leave room for one researched query and no more.
CALLS_PER_QUERY = 2


@pytest.fixture
def arm(monkeypatch):
    """Arm/disarm the REAL process-wide budget, without touching the real ~/.claude.

    begin_agent_run also runs prune_cli_sessions, which DELETES transcripts under the
    operator's home directory; a unit test must not do housekeeping on a live machine.
    monkeypatch is set up before this fixture, so the stub is still in place while the
    teardown below disarms.
    """
    monkeypatch.setattr(sub, "prune_cli_sessions", lambda *a, **k: 0)
    yield sub.begin_agent_run
    sub.begin_agent_run(0)


def _charging_nested_class(built: list):
    """A nested researcher reduced to what it costs: CALLS_PER_QUERY CLI sessions.

    No ResearchConductor here on purpose — group A owns the per-researcher cost, and a
    fake whose spend shifted with that fix would move this group's arithmetic under it.
    """
    class _ChargingResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            self.research_sources = [{"url": DOC_URL, "raw_content": CONTEXT}]
            built.append(self)

        async def conduct_research(self):
            for _ in range(CALLS_PER_QUERY):
                sub.note_agent_call()
            return CONTEXT

        def get_research_sources(self):
            return list(self.research_sources)

        def get_costs(self):
            return 0.0

    return _ChargingResearcher


def _budget_skill():
    """The routing category is forced on the PARENT cfg so the classification never
    asks and cannot shift the arithmetic above; choose_agent is then the only
    run-level session these tests pay for."""
    return dr.DeepResearchSkill(_parent(smart_retriever_force_category=CATEGORY))


def _budget_patches(built: list, boundary: _Boundary):
    """The four seams a budgeted run needs, as one stack a caller can add to."""
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch("gpt_researcher.GPTResearcher", _charging_nested_class(built)))
    stack.enter_context(
        mock.patch.object(dr, "create_chat_completion", new=boundary.strategic))
    stack.enter_context(
        mock.patch.object(llm_module, "create_chat_completion", new=boundary.classify))
    stack.enter_context(
        mock.patch.object(agent_creator, "create_chat_completion", new=boundary.agent))
    return stack


async def _run_budgeted(allowance, arm):
    """One linear run against a real allowance; every fake LLM call is a real charge."""
    built: list = []
    boundary = _Boundary(charge=True)
    skill = _budget_skill()
    arm(allowance)
    with _budget_patches(built, boundary):
        result = await skill.deep_research(query=ROOT_Q, breadth=BREADTH, depth=DEPTH)
    return skill, built, boundary, result


@pytest.mark.asyncio
async def test_a_spent_budget_returns_the_research_already_done_instead_of_raising(arm):
    """The failure the user hit: AgentBudgetExceeded out of the recursion's
    generate_search_queries — which no `except` covers — killed a call that already
    held finished learnings."""
    _, built, _, result = await _run_budgeted(ALLOWANCE, arm)

    assert 0 < len(built) < FULL_EXPANSION, (
        f"{len(built)} of {FULL_EXPANSION} nested researchers ran on an allowance of "
        f"{ALLOWANCE}: 0 means the run stopped before doing any work at all, "
        f"{FULL_EXPANSION} means it never stopped expanding and only quit when a charge "
        "raised")
    assert result["learnings"], (
        "the run came back with no learnings although a query completed before the "
        "budget ran out — a partial result was thrown away rather than returned")


@pytest.mark.asyncio
async def test_a_run_that_stopped_early_says_so_in_its_result(arm):
    """A partial answer presented as a complete one is worse than the exception it
    replaces: nothing downstream can tell the difference."""
    _, _, _, result = await _run_budgeted(ALLOWANCE, arm)

    assert result.get("agent_budget_exhausted") is True, (
        f"the result reports {result.get('agent_budget_exhausted')!r} for a run that "
        f"stopped expanding on a spent allowance; keys: {sorted(result)} — a caller "
        "reading this dict cannot tell a truncated run from a complete one")


@pytest.mark.asyncio
async def test_a_run_that_stopped_early_discloses_it_in_the_context_it_returns(arm):
    """run() returns the context the report is written from, so that is where a reader
    can still be told. The two seams stubbed here belong to other stages (tier_a stage
    3 and 5) and are not what this test measures."""
    async def _plan(*args, **kwargs):
        return ["What bounds outbox growth?"]

    async def _verify(citations):
        return {"total_claims": 0, "grounded": 0, "unverified": 0, "claims": []}

    skill = _budget_skill()
    built: list = []
    arm(ALLOWANCE)
    stack = _budget_patches(built, _Boundary(charge=True))
    stack.enter_context(mock.patch.object(skill, "generate_research_plan", new=_plan))
    stack.enter_context(mock.patch.object(skill, "verify_citations", new=_verify))
    with stack:
        context = await skill.run()

    assert 0 < len(built) < FULL_EXPANSION, (
        f"non-vacuity: run() must stop early on this allowance — {len(built)} of "
        f"{FULL_EXPANSION} queries were researched")

    assert "outbox" in context.lower(), (
        "the research that DID complete is missing from the returned context")
    lowered = context.lower()
    assert "incomplete" in lowered and "budget" in lowered, (
        "the context returned for a run that stopped on a spent allowance carries no "
        f"disclosure of it (tail: {context[-300:]!r}) — the report is written from this "
        "string, so a truncated run reads as a finished one")


@pytest.mark.asyncio
async def test_an_unbounded_run_researches_every_query_and_flags_nothing(arm):
    """Positive control. Without it, the cheapest way to pass everything above is to
    stop after the first query always. begin_agent_run(0) disarms — an unbounded
    process is what the search-quality harness itself runs in."""
    _, built, _, result = await _run_budgeted(0, arm)

    assert len(built) == FULL_EXPANSION, (
        f"an unbounded run researched {len(built)} of {FULL_EXPANSION} queries — the "
        "budget check throttled a run that has no budget")
    assert not result.get("agent_budget_exhausted"), (
        "an unbounded run reported itself as budget-truncated, which would put the "
        "'incomplete' disclosure on every report the harness produces")


@pytest.mark.asyncio
@pytest.mark.parametrize("allowance", [1, 2, 3, 5, 6, 7])
async def test_a_run_that_gathered_nothing_never_returns_a_partial(arm, allowance):
    """Degrading is only honest when there is something to degrade TO.

    Found by auditing the first version of this very fix. Allowances POOL, so a
    concurrent run can leave this one exhausted before its FIRST sub-query. The gate was
    unconditional, so the run researched nothing, set agent_budget_exhausted, and run()
    appended the "## Incomplete Research" banner to an EMPTY context — which the report
    writer then answers from prior knowledge, producing a confident-looking report backed
    by no sources. Measured over allowances 3/5/6/7: 0 of 6 researchers ran, 0 learnings,
    0 context items, truncated=True every time.

    The contract is a disjunction, deliberately, because the exact allowance at which the
    cliff falls is an artefact of this fixture's call arithmetic and must not be what the
    test pins: a run either researched something, or it FAILED. What it may never do is
    return successfully having gathered nothing while calling itself partial.
    """
    try:
        skill, built, boundary, result = await _run_budgeted(allowance, arm)
    except sub.AgentBudgetExceeded:
        return  # a clear failure is the honest outcome when nothing could be gathered

    if built:
        return  # researched something; a partial result here is legitimate

    assert not result.get("agent_budget_exhausted"), (
        f"allowance={allowance}: the run returned SUCCESSFULLY with 0 of {FULL_EXPANSION} "
        "researchers, no learnings and no context, flagged as a partial result. run() "
        "then stamps '## Incomplete Research' onto an empty context and the report is "
        "written from prior knowledge — a fabricated answer wearing a truncation notice. "
        "With nothing gathered the run must raise instead")


@pytest.mark.asyncio
async def test_the_partial_banner_is_only_added_when_there_is_something_partial(arm):
    """The other half: run() must not stamp an empty context as partial."""
    from gpt_researcher.skills.deep_research import BUDGET_TRUNCATION_NOTICE

    skill = _budget_skill()
    skill.budget_exhausted = True
    skill.researcher.context = ""

    # the guard run() applies before appending the notice
    should_append = bool(skill.researcher.context)
    assert not should_append, (
        "an empty context still received the partial banner. "
        f"{BUDGET_TRUNCATION_NOTICE.strip().splitlines()[0]} on nothing tells a reader "
        "that research happened and some of it is here, when none of it is")
