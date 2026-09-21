"""s21: the LINEAR deep-research path is bounded by wall clock, not only by calls.

Measured on the MCP container 2026-09-20, breadth=3/depth=2, the shipped defaults:
one run took 23m31s end to end, a second was still running at 43m. Nothing in the path
was broken — `begin_agent_run` bounds what a run may SPEND and every gate in
`deep_research()` reads that counter, but no gate reads a clock, and the two costs are
not the same cost. What made the difference was time nobody was billed for:

  13:39:17  run() starts
  13:39:57  3 top-level queries planned          (40s)
  13:47:51  the depth-1 round finishes           (~8m, 3 nested researchers in parallel)
  13:52:09 / 13:55:29 / 14:02:30  the three depth-2 recursions, ONE AFTER ANOTHER
  14:02:48  citation verification                (18s)

The recursions sit inside `for result in results:`, so they are sequential: the 8-minute
round that produced them cost three times as much again. And underneath, a single
sub-query blocked 5 minutes inside one embedding call (23:05:50 retry -> 23:10:51
APITimeoutError, exactly the 300s `request_timeout` in EMBEDDING_KWARGS), which no gate
between sub-queries can shorten — only a timeout on the sub-query itself.

So this file pins three things about `DEEP_RESEARCH_TIME_BUDGET_S`:
  - a spent budget costs the run its DEPTH, not its result;
  - the run says so, in the result dict AND in the context the report is written from;
  - a run that has gathered NOTHING is never returned as a partial (the s11 lesson,
    re-derived here: an empty context wearing a truncation banner is answered from
    prior knowledge by the report writer).

Deterministic: no network, no live LLM, no container, no real sleeping. The clock is
`time.monotonic`, stubbed to advance by a fixed amount per nested researcher, so "slow"
is a number this file chooses rather than wall time it waits out.
"""
from __future__ import annotations

import asyncio
import contextlib
import time as _real_time
from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.deep_research as dr
import gpt_researcher.utils.llm as llm_module
from gpt_researcher.actions import agent_creator
from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.prompts import PromptFamily

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
DOC_URL = "https://example.invalid/outbox"
CONTEXT = "The outbox table grows without bound unless a cleanup job trims it. " * 40

LLM_BLOB = (
    "Query: how production teams bound outbox table growth\n"
    "Goal: find retention practice\n"
    "Query: which brokers deduplicate replayed outbox messages\n"
    "Goal: find dedup guarantees\n"
    f"Learning [{DOC_URL}]: The outbox table grows without bound unless a job trims it.\n"
    "Question: What trims the outbox table in practice?\n"
)

# breadth=2, depth=2 -> 2 top-level queries, each recursing into 2 more: 6 nested
# researchers if nothing stops the expansion.
BREADTH, DEPTH = 2, 2
FULL_EXPANSION = 6

# What one nested researcher "takes". concurrency=1 so the fake clock is a straight
# sum rather than something that depends on scheduling order.
SECONDS_PER_QUERY = 100.0


class _Clock:
    """A monotonic clock this file advances explicitly.

    Patched onto the module rather than slept through: the point is to measure the
    gate, and a test that actually waited 600 seconds measures the CI machine.
    """

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _cfg(**over):
    base = dict(
        fast_llm_provider="fake", fast_llm_model="fast-model",
        strategic_llm_provider="fake", strategic_llm_model="strategic-model",
        smart_llm_provider="fake", smart_llm_model="smart-model",
        llm_kwargs={}, reasoning_effort="low", curate_sources=False,
        smart_retriever_config=None, smart_retriever_force_category="academic",
        config_path=None, deep_research_breadth=BREADTH, deep_research_depth=DEPTH,
        deep_research_concurrency=1, max_iterations=1, max_search_results_per_query=5,
        deep_research_time_budget_s=600.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _parent(**cfg_over):
    return SimpleNamespace(
        query=ROOT_Q, cfg=_cfg(**cfg_over), tone=None, websocket=None, headers={},
        visited_urls=set(), mcp_configs=None, mcp_strategy=None, research_sources=[],
        parent_query="", prompt_family=PromptFamily, retrievers=[], log_handler=None,
        context="", add_costs=lambda *a, **k: None, get_costs=lambda: 0.0,
    )


async def _fake_llm(*args, **kwargs):
    return LLM_BLOB


def _timed_nested_class(built: list, clock: _Clock, seconds=SECONDS_PER_QUERY):
    """A nested researcher reduced to what it COSTS IN TIME.

    No ResearchConductor and no charging: s11 owns per-researcher session cost, and a
    fake that also spent budget would let the budget gate decide these tests.
    """
    class _SlowResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            self.research_sources = [{"url": DOC_URL, "raw_content": CONTEXT}]
            built.append(self)

        async def conduct_research(self):
            clock.advance(seconds)
            return CONTEXT

        def get_research_sources(self):
            return list(self.research_sources)

        def get_costs(self):
            return 0.0

    return _SlowResearcher


@pytest.fixture
def unbudgeted(monkeypatch):
    """Disarm the CLI-session allowance so only the clock can stop these runs.

    Patched where it LIVES: deep_research imports it inside the function, so the
    attribute is resolved at call time.
    """
    monkeypatch.setattr(sub, "agent_budget_exhausted", lambda reserve=0: False)
    monkeypatch.setattr(sub, "research_should_stop", lambda reserve=0: False)


def _patches(built, clock):
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch("gpt_researcher.GPTResearcher",
                                   _timed_nested_class(built, clock)))
    stack.enter_context(mock.patch.object(dr, "create_chat_completion", new=_fake_llm))
    stack.enter_context(mock.patch.object(llm_module, "create_chat_completion", new=_fake_llm))
    stack.enter_context(mock.patch.object(agent_creator, "create_chat_completion", new=_fake_llm))
    # The module's `time` NAME is replaced, not `time.monotonic` itself: dr.time IS the
    # stdlib module, and asyncio's event loop reads time.monotonic too — freezing it
    # there hangs every await on a timer (found the hard way: a 10-minute pytest hang).
    stack.enter_context(mock.patch.object(
        dr, "time", SimpleNamespace(monotonic=clock, time=_real_time.time)))
    return stack


async def _run(budget_s, plan_seconds=0.0):
    """One hermetic linear run() against a fake clock. Returns (skill, nested, context).

    `plan_seconds` is what generate_research_plan "takes": it runs INSIDE the budget
    (run() starts the clock first), so it is how a test spends the whole budget before
    the first sub-query is ever reached."""
    built: list = []
    clock = _Clock()
    skill = dr.DeepResearchSkill(_parent(deep_research_time_budget_s=budget_s))

    async def _plan(*args, **kwargs):
        clock.advance(plan_seconds)
        return ["What bounds outbox growth?"]

    async def _verify(citations):
        return {"total_claims": 0, "grounded": 0, "unverified": 0, "claims": []}

    stack = _patches(built, clock)
    stack.enter_context(mock.patch.object(skill, "generate_research_plan", new=_plan))
    stack.enter_context(mock.patch.object(skill, "verify_citations", new=_verify))
    with stack:
        context = await skill.run()
    return skill, built, context


# A budget that the 2 top-level queries fit inside (2 x 100s) but the depth-2
# recursions do not. Deliberately not a round number of SECONDS_PER_QUERY: the gate
# must read the clock, not count researchers.
TIGHT_BUDGET = 250.0


@pytest.mark.asyncio
async def test_a_spent_time_budget_costs_the_run_its_depth_not_its_result(unbudgeted):
    """The 23m31s run in the docstring spent 14 of those minutes on recursions that
    ran one after another. Stopping is only useful if what already finished survives."""
    skill, built, context = await _run(TIGHT_BUDGET)

    assert 0 < len(built) < FULL_EXPANSION, (
        f"{len(built)} of {FULL_EXPANSION} nested researchers ran inside a "
        f"{TIGHT_BUDGET:.0f}s budget at {SECONDS_PER_QUERY:.0f}s each: 0 means the run "
        f"stopped before doing any work, {FULL_EXPANSION} means the clock never stopped "
        "it and a 10-minute budget is decoration")
    assert "outbox" in context.lower(), (
        "the research that DID finish inside the budget is missing from the returned "
        "context — the run threw away work it had already paid for")


@pytest.mark.asyncio
async def test_a_time_truncated_run_says_so_in_its_result_and_in_its_context(unbudgeted):
    """Both places, for the reason BUDGET_TRUNCATION_NOTICE already documents: the
    dict is what the MCP caller reads, the context is what the report is WRITTEN from,
    and a partial answer presented as a complete one is worse than an error."""
    skill, built, context = await _run(TIGHT_BUDGET)

    assert skill.time_exhausted is True, (
        f"skill.time_exhausted is {skill.time_exhausted!r} for a run that stopped "
        "expanding on the clock — nothing downstream can tell it from a complete run")
    lowered = context.lower()
    assert "incomplete" in lowered and "time budget" in lowered, (
        "the context returned for a time-truncated run carries no disclosure of it "
        f"(tail: {context[-300:]!r}) — the report is written from this string, so a "
        "run that answered half the questions reads as one that answered all of them")


@pytest.mark.asyncio
async def test_an_unbounded_run_expands_fully_and_flags_nothing(unbudgeted):
    """Positive control. Without it the cheapest way to pass everything above is to
    stop after the first query always. 0 is the documented 'no clock' value and is
    what the search-quality harness runs, where a full expansion IS the measurement."""
    skill, built, context = await _run(0)

    assert len(built) == FULL_EXPANSION, (
        f"an unbounded run researched {len(built)} of {FULL_EXPANSION} queries — the "
        "clock throttled a run that was told it has none")
    assert not skill.time_exhausted, (
        "an unbounded run reported itself as time-truncated, which would put the "
        "'Incomplete Research' banner on every report the harness produces")
    assert "incomplete" not in context.lower(), (
        "a complete run's context carries a truncation banner")


@pytest.mark.asyncio
async def test_a_run_that_gathered_nothing_is_never_returned_as_a_partial(unbudgeted):
    """The s11 lesson, re-derived for the clock: degrading is only honest when there is
    something to degrade TO. A budget already spent when the first sub-query starts
    must not produce an empty context stamped 'Incomplete Research' — the report writer
    answers that from prior knowledge and it looks like sourced research."""
    # A real, positive budget that planning alone overspends -- so every gate sees an
    # expired deadline. (A budget <= 0 would not do: that is the "unbounded" value.)
    skill, built, context = await _run(10.0, plan_seconds=60.0)
    assert skill.time_exhausted, "non-vacuity: the gates must have seen an expired deadline"

    assert built, (
        "the first batch was gated on a deadline that was already expired, so the run "
        "researched nothing at all; the first query must always be allowed to run")
    assert context.strip(), "a run that researched something returned an empty context"


@pytest.mark.asyncio
async def test_one_slow_sub_query_cannot_outlive_the_budget(unbudgeted, monkeypatch):
    """What the gates alone cannot give. Measured 2026-09-20: one sub-query sat ~5
    minutes inside a single embedding call before the 300s request_timeout fired. A
    check between sub-queries never runs while that one is in flight, so
    conduct_research() itself is what has to be bounded.

    Real clock here, not the fake one: asyncio.wait_for times out on the LOOP's clock.
    The floor is shrunk so the test waits a fraction of a second, not 30."""
    monkeypatch.setattr(dr, "MIN_SUB_QUERY_TIMEOUT_S", 0.05)
    built: list = []

    class _NeverReturns:
        def __init__(self, query=None, **kwargs):
            built.append(self)
            self.visited_urls = set()
            self.research_sources = []
            self.cfg = _cfg()

        async def conduct_research(self):
            await asyncio.sleep(3600)

        def get_research_sources(self):
            return []

        def get_costs(self):
            return 0.0

    skill = dr.DeepResearchSkill(_parent())
    skill._deadline = _real_time.monotonic() + 0.05
    with mock.patch("gpt_researcher.GPTResearcher", _NeverReturns), \
         mock.patch.object(dr, "create_chat_completion", new=_fake_llm), \
         mock.patch.object(llm_module, "create_chat_completion", new=_fake_llm), \
         mock.patch.object(agent_creator, "create_chat_completion", new=_fake_llm):
        result = await asyncio.wait_for(
            skill.deep_research(query=ROOT_Q, breadth=1, depth=1), timeout=10)

    assert built, "non-vacuity: the sub-query has to have been started"
    assert result["learnings"] == [] and not result["context"], (
        "a sub-query that never returns produced a result — it was not cancelled")
    # depth=1 is the shipped default, so there is no recursion gate after this batch
    # to notice what the timeout dropped. The cancellation itself has to say so.
    assert skill.time_exhausted is True and result["time_budget_exhausted"] is True, (
        "a sub-query was cancelled at the deadline and the run did not mark itself "
        "partial — the caller gets 2 of 3 questions answered, presented as complete")
