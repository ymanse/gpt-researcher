"""s11 (P1.2): choosing the agent is a per-RUN decision, not a per-node one.

Measured 2026-09-05 (harness-search/spec/p0-p1-slimming.md): one tree node costs
8-9 `claude` CLI sessions against a 100-session cap, and `choose_agent` is one of
them. Every node builds its own GPTResearcher, every one of them arrives with
`agent=None, role=None`, so `ResearchConductor.conduct_research` re-selects an
agent for the same run once per node. 20 nodes spend 20 sessions on a decision
whose answer never changes, and the tree is cut short by its own overhead.

WHAT IS COUNTED. The counter below is the LLM boundary INSIDE `choose_agent`
(`gpt_researcher.actions.agent_creator.create_chat_completion`), not the name
`choose_agent` in whichever module happens to import it. The budget's unit is
sessions spawned, so that is the unit measured here; how the fix reaches (or
avoids) that function is not part of the contract.

THE TRAP. The guard is `if not (self.researcher.agent and self.researcher.role)`
-- an AND over two truthy checks. Handing a node researcher an agent but no role,
or an empty string for either, buys nothing: the guard still fires and the session
is still spent. Two tests here pin that as it stands today, so the slimming cannot
be "achieved" by loosening the guard to an OR -- that would research every node
under a role prompt nobody chose.

Deterministic: no network, no real LLM, no embeddings service. The web search is
faked at the ResearchConductor class level and both LLM entry points are AsyncMocks.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.tree_research as tr
import gpt_researcher.utils.llm as llm_module
from gpt_researcher import GPTResearcher
from gpt_researcher.actions import agent_creator
from gpt_researcher.prompts import PromptFamily
from gpt_researcher.skills.researcher import ResearchConductor

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
C1 = "What does the Debezium documentation say about outbox table growth?"
C2 = "Which managed change-data-capture vendors publish outbox latency figures?"

# orthogonal, so nothing here is ever pruned for near-duplication (that is s4/s8
# territory, not this file's)
EMB = {ROOT_Q: [1.0, 0.0, 0.0], C1: [0.0, 1.0, 0.0], C2: [0.0, 0.0, 1.0]}

AGENT = "Data Engineering Agent"
ROLE = "You are a data engineering researcher. Answer with sourced specifics."

SOURCE_URL = "https://example.invalid/outbox"
# research_node fails a node closed under MIN_CONTEXT_CHARS (8000) and run() then
# never expands it, so the fake context has to be a realistic size -- otherwise the
# "N nodes" claim below would quietly be measured on N == 1.
CONTEXT = "The outbox table grows without bound unless a cleanup job trims it. " * 140

ANSWER_BLOB = (
    "ANSWER: The outbox table grows without bound unless a cleanup job trims it.\n"
    "DIGEST: Outbox tables need a cleanup job.\n"
    "LEARNINGS:\n- The outbox table grows without bound unless a cleanup job trims it.\n"
)


class _FakeRetriever:
    """conduct_research reads `r.__name__` off every configured retriever."""


def _agent_llm() -> mock.AsyncMock:
    """The single call `choose_agent` makes; each invocation is one CLI session."""
    return mock.AsyncMock(return_value=json.dumps(
        {"server": AGENT, "agent_role_prompt": ROLE}))


def _node_researcher_class(built: list):
    """Stand-in for the per-node GPTResearcher that keeps the REAL guard.

    It records how TreeResearchSkill constructed it and then runs the shipped
    `ResearchConductor.conduct_research` over itself, so `choose_agent` fires -- or
    does not -- for exactly the reason the shipped code makes it fire. Nothing about
    the guard is re-implemented here; only the web search and the LLM are faked.
    """
    class _NodeResearcher:
        def __init__(self, query=None, **kwargs):
            built.append({"query": query, **kwargs})
            self.query = query
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
            self.cfg = SimpleNamespace(
                smart_llm_model="fake-smart", smart_llm_provider="fake",
                llm_kwargs={}, curate_sources=False)
            self.research_conductor = ResearchConductor(self)

        def add_costs(self, *args, **kwargs):
            pass

        def get_costs(self):
            return 0.0

        async def conduct_research(self):
            return await self.research_conductor.conduct_research()

        def get_research_sources(self):
            return [{"url": SOURCE_URL, "raw_content": CONTEXT}]

    return _NodeResearcher


@pytest.fixture(scope="module")
def tree_parent():
    """A real GPTResearcher as the tree's parent.

    Not a namespace: resolving (agent, role) once for the run needs the parent's
    cfg / prompt_family / add_costs, and a namespace missing any of them sends
    choose_agent down its `except Exception` path to a default agent -- which would
    decide these tests for a fixture reason instead of a behavioural one.
    """
    return GPTResearcher(query=ROOT_Q, verbose=False)


async def _run_tree(parent, tmp_path):
    """One three-node run (root + two children) with every LLM/web seam faked."""
    built: list = []
    agent_llm = _agent_llm()
    skill = tr.TreeResearchSkill(parent)
    skill.visited_urls.clear()

    async def _embed(text):
        return list(EMB.get(text, [0.0, 0.0, 0.0]))

    async def _children(node):
        return [C1, C2] if node.depth == 0 else []

    skill.embed_question = _embed
    # HERMETIC TRIPWIRE, closed before the feature that needs it arrives.
    # `_classify_query` resolves `create_chat_completion` out of gpt_researcher.utils.llm
    # at CALL time, and no patch below covers that binding. The day P1.1 lands and
    # TreeResearchSkill resolves the run's routing category through it, this fixture stops
    # being hermetic — and on a machine where the claude_agent CLI is live it does not
    # raise, it SUCCEEDS: real `claude` sessions spawned per suite run, charged to the
    # subscription, with the test passing either way. Silent is the problem, not slow.
    classify_llm = mock.AsyncMock(return_value="general_web")
    with mock.patch.object(tr, "GPTResearcher", _node_researcher_class(built)), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children), \
         mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value=ANSWER_BLOB)), \
         mock.patch.object(llm_module, "create_chat_completion", new=classify_llm), \
         mock.patch.object(ResearchConductor, "_get_context_by_web_search",
                           new=mock.AsyncMock(return_value=[CONTEXT])), \
         mock.patch.object(agent_creator, "create_chat_completion", new=agent_llm):
        result = await skill.run(query=ROOT_Q, max_depth=1, max_breadth=2,
                                 max_nodes=10, outputs_dir=str(tmp_path))
    return result, built, agent_llm


# --------------------------------------------------------------- the guard itself
# Regression guards: this is what conduct_research does today, and the slimming
# must not reach its call count by weakening it.

@pytest.mark.asyncio
async def test_a_researcher_handed_both_agent_and_role_spends_no_session_choosing_one():
    """The whole premise of P1.2: a fully specified researcher skips the selection."""
    researcher = GPTResearcher(query=ROOT_Q, agent=AGENT, role=ROLE, verbose=False)
    agent_llm = _agent_llm()
    with mock.patch.object(agent_creator, "create_chat_completion", new=agent_llm), \
         mock.patch.object(ResearchConductor, "_get_context_by_web_search",
                           new=mock.AsyncMock(return_value=[CONTEXT])):
        await researcher.conduct_research()

    assert agent_llm.call_count == 0, (
        f"a researcher given both agent and role still spent {agent_llm.call_count} "
        "CLI session(s) re-choosing them; that is the per-node cost P1.2 removes"
    )
    assert (researcher.agent, researcher.role) == (AGENT, ROLE), (
        "the caller's agent/role were overwritten by a selection nobody asked for"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent,role", [
    (AGENT, None),
    (None, ROLE),
    ("", ROLE),
    (AGENT, ""),
], ids=["agent-only", "role-only", "empty-agent", "empty-role"])
async def test_a_half_specified_researcher_still_pays_for_the_selection(agent, role):
    """The AND is the trap. Passing one half of the pair (or an empty string for
    either half) saves nothing -- the session is still spent and BOTH halves are
    replaced. A caller that hands its nodes only `agent` has slimmed nothing."""
    researcher = GPTResearcher(query=ROOT_Q, agent=agent, role=role, verbose=False)
    agent_llm = _agent_llm()
    with mock.patch.object(agent_creator, "create_chat_completion", new=agent_llm), \
         mock.patch.object(ResearchConductor, "_get_context_by_web_search",
                           new=mock.AsyncMock(return_value=[CONTEXT])):
        await researcher.conduct_research()

    assert agent_llm.call_count == 1, (
        f"agent={agent!r} role={role!r} must not satisfy the guard, but the "
        f"selection ran {agent_llm.call_count} time(s) -- half a pair is not a "
        "resolved agent, and skipping the call would leave the run researching "
        "under a role prompt nobody chose"
    )
    assert (researcher.agent, researcher.role) == (AGENT, ROLE), (
        f"the half-specified pair survived as {(researcher.agent, researcher.role)!r}; "
        "the guard replaces both halves or neither"
    )


# ------------------------------------------------------- once per run, not per node

@pytest.mark.asyncio
async def test_every_node_researcher_is_handed_the_same_resolved_agent_and_role(
        tree_parent, tmp_path):
    """P1.2: TreeResearchSkill resolves (agent, role) for the run and hands BOTH to
    every per-node GPTResearcher, so no node re-decides it."""
    result, built, _ = await _run_tree(tree_parent, tmp_path)

    assert result["stats"]["researched"] == 3, (
        "fixture precondition: the run must research 3 nodes, got "
        f"{result['stats']['researched']}"
    )
    # only the per-node researchers are the contract; an implementation free to
    # build something else on the side must not fail here for that reason
    node_builds = [kw for kw in built if kw["query"] in (ROOT_Q, C1, C2)]
    assert len(node_builds) == 3, (
        f"one researcher per node expected, got {len(node_builds)}"
    )

    unresolved = [kw for kw in node_builds if not (kw.get("agent") and kw.get("role"))]
    assert not unresolved, (
        f"{len(unresolved)} of {len(node_builds)} node researchers were built without "
        "a resolved (agent, role) pair, so each of them re-runs the selection itself: "
        f"{[(kw.get('agent'), kw.get('role')) for kw in node_builds]}"
    )
    pairs = {(kw.get("agent"), kw.get("role")) for kw in node_builds}
    assert len(pairs) == 1, (
        f"the nodes of one run were given {len(pairs)} different (agent, role) pairs "
        f"({pairs}); the selection is a property of the run, not of the node"
    )


@pytest.mark.asyncio
async def test_a_three_node_run_spends_one_selection_session_not_one_per_node(
        tree_parent, tmp_path):
    """The saving, measured the way the budget measures it: CLI sessions spawned.

    `<= 1` and not `== 1` on purpose -- resolving the pair without an LLM at all
    also satisfies the spec; what may never happen is the count scaling with the
    number of nodes.
    """
    result, built, agent_llm = await _run_tree(tree_parent, tmp_path)

    assert result["stats"]["researched"] == 3, (
        "fixture precondition: the run must research 3 nodes, got "
        f"{result['stats']['researched']}"
    )
    assert agent_llm.call_count <= 1, (
        f"{result['stats']['researched']} researched nodes cost "
        f"{agent_llm.call_count} agent-selection CLI sessions -- the count scales "
        "with the tree instead of being paid once per run, which is what eats a "
        "100-session budget at 20 nodes"
    )
