"""s22: stage 2 runs CONCURRENTLY.

Measured on the MCP container 2026-09-20 at the shipped defaults (breadth=3, depth=2),
in the 23m31s run:

  13:39:57  3 top-level queries planned
  13:47:51  the stage-1 round finishes          ~8m, 3 nested researchers IN PARALLEL
  13:52:09 / 13:55:29 / 14:02:30                 the 3 stage-2 recursions, ONE AT A TIME
  14:02:48  citation verification

The recursion sat inside `for result in results:`, so sibling follow-up rounds ran
sequentially although each reads only its own result: 8 minutes of parallel work produced
14 minutes of serial work.

So this file pins:
  - stage 2 is dispatched at once, bounded by the RUN's semaphore (not one per level);
  - a failed branch costs its own findings, never its siblings';
  - each stage-2 branch PLANS against what every sibling already found, and may plan
    nothing when the root query is answered -- the one place a round can shrink;
  - branches plan ONE AFTER ANOTHER, each shown what earlier siblings already planned,
    so they do not all converge on the same gap -- then research concurrently.

A judge that filtered these follow-ups lived here briefly and was removed on 2026-09-21:
it could only save WHOLE branches (`new_breadth` does not shrink with the number of
surviving questions), and a live run kept 20 of 22 candidates with 0 branches dropped, so
it cost an LLM call per level and bought nothing. See s22's git history.

Deterministic: no network, no live LLM, no container. Concurrency is observed by
recording enter/exit around a real `asyncio.sleep(0)` rather than by wall time.
"""
from __future__ import annotations

import asyncio
import contextlib
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

BREADTH, DEPTH = 2, 2
# 2 stage-1 queries, each recursing into a round of 2: 6 nested researchers when the
# judge keeps everything.
FULL_EXPANSION = 6

# One follow-up per stage-1 branch.
FOLLOW_UPS = [
    "What trims the outbox table in practice?",
    "How do brokers deduplicate replayed outbox messages?",
]

# The two stage-1 queries, keyed by a word that appears in exactly one of them so the
# fake LLM can tell which branch is asking.
QUERY_BLOB = (
    "Query: how production teams bound outbox table growth\n"
    "Goal: find retention practice\n"
    "Query: which brokers deduplicate replayed outbox messages\n"
    "Goal: find dedup guarantees\n"
)


# What each stage-1 branch finds. Distinct per branch, so a stage-2 planning prompt that
# carries only its OWN branch's findings is distinguishable from one that carries all.
STAGE1 = [
    ("how production teams bound outbox table growth",
     "Teams partition the outbox by day and drop old partitions."),
    ("which brokers deduplicate replayed outbox messages",
     "Consumers deduplicate replays with an inbox table keyed by message id."),
]


def _answer_blob(prompt: str) -> str:
    """Learnings plus the follow-up belonging to whichever branch is answering."""
    dedup = "deduplicate" in prompt
    learning = STAGE1[1][1] if dedup else STAGE1[0][1]
    question = FOLLOW_UPS[1] if dedup else FOLLOW_UPS[0]
    return f"Learning [{DOC_URL}]: {learning}\nQuestion: {question}\n"


def _cfg(**over):
    base = dict(
        fast_llm_provider="fake", fast_llm_model="fast-model",
        strategic_llm_provider="fake", strategic_llm_model="strategic-model",
        smart_llm_provider="fake", smart_llm_model="smart-model",
        merge_llm_provider=None, merge_llm_model=None,
        llm_kwargs={}, reasoning_effort="low", curate_sources=False,
        smart_retriever_config=None, smart_retriever_force_category="academic",
        config_path=None, deep_research_breadth=BREADTH, deep_research_depth=DEPTH,
        deep_research_concurrency=4, max_iterations=1, max_search_results_per_query=5,
        deep_research_time_budget_s=0.0,  # unbounded: s21 owns the clock
        embedding_provider="fake", embedding_model="fake-embed", embedding_kwargs={},
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


class _Trace:
    """Every nested researcher's overlap window, and the judge calls."""

    def __init__(self):
        self.built: list = []
        self.live = 0
        self.max_live = 0
        self.fail_branch: str | None = None
        self.failed = 0
        self.completed: list = []
        self.plan_prompts: list = []
        # What stage-2 PLANNING returns. None = two queries unique to that planning call
        # ("stage2 plan <n> a/b"), so a later branch's prompt can be checked for an
        # earlier branch's plan; "" = the planner judged the root answered.
        self.stage2_plan: str | None = None
        self.stage2_plans_made = 0
        # 1-based index of the stage-2 planning call that raises, or None.
        self.fail_stage2_plan: int | None = None

    async def llm(self, *args, **kwargs):
        """One fake for every call site, told apart by what the prompt asks for."""
        messages = kwargs.get("messages") or []
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "key learnings" in text:
            return _answer_blob(text)
        if "search queries" in text:
            self.plan_prompts.append(text)
            if "ALREADY covered" in text:
                self.stage2_plans_made += 1
                n = self.stage2_plans_made
                if self.fail_stage2_plan == n:
                    raise RuntimeError(f"stage-2 planning call {n} failed")
                if self.stage2_plan is not None:
                    return self.stage2_plan
                return (f"Query: stage2 plan {n} a\nGoal: goal {n}a\n"
                        f"Query: stage2 plan {n} b\nGoal: goal {n}b\n")
        return QUERY_BLOB


def _nested_class(trace: _Trace):
    class _Researcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            self.research_sources = [{"url": DOC_URL, "raw_content": CONTEXT}]
            trace.built.append(self)

        async def conduct_research(self):
            trace.live += 1
            trace.max_live = max(trace.max_live, trace.live)
            try:
                # A real suspension point: without one the coroutines would run to
                # completion in dispatch order and `max_live` would read 1 even when
                # the calls ARE concurrent.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                if trace.fail_branch and trace.fail_branch in str(self.query):
                    trace.failed += 1
                    raise RuntimeError("branch exploded")
                trace.completed.append(str(self.query))
                return CONTEXT
            finally:
                trace.live -= 1

        def get_research_sources(self):
            return list(self.research_sources)

        def get_costs(self):
            return 0.0

    return _Researcher


@pytest.fixture
def unbudgeted(monkeypatch):
    monkeypatch.setattr(sub, "agent_budget_exhausted", lambda reserve=0: False)
    monkeypatch.setattr(sub, "research_should_stop", lambda reserve=0: False)


def _patches(trace: _Trace):
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch("gpt_researcher.GPTResearcher", _nested_class(trace)))
    stack.enter_context(mock.patch.object(dr, "create_chat_completion", new=trace.llm))
    stack.enter_context(mock.patch.object(llm_module, "create_chat_completion", new=trace.llm))
    stack.enter_context(mock.patch.object(agent_creator, "create_chat_completion", new=trace.llm))
    return stack


async def _run(trace: _Trace, **cfg_over):
    skill = dr.DeepResearchSkill(_parent(**cfg_over))
    with _patches(trace):
        result = await skill.deep_research(query=ROOT_Q, breadth=BREADTH, depth=DEPTH)
    return skill, result


# --------------------------------------------------------------------------- A: parallel

@pytest.mark.asyncio
async def test_stage_two_branches_run_at_the_same_time(unbudgeted):
    """The defect: sibling follow-up rounds ran one after another.

    Counted by overlap, not by clock: `max_live` is how many nested researchers were
    inside conduct_research simultaneously. Stage 1 alone reaches 2, so the bar is
    ABOVE that -- a sequential stage 2 would leave it at exactly 2.
    """
    trace = _Trace()
    _, result = await _run(trace)

    assert len(trace.built) == FULL_EXPANSION, (
        f"fixture precondition: {FULL_EXPANSION} nested researchers with the judge "
        f"keeping everything, got {len(trace.built)}")
    assert trace.max_live > BREADTH, (
        f"at most {trace.max_live} nested researchers were ever in flight at once. "
        f"Stage 1 alone accounts for {BREADTH}, so the stage-2 rounds still ran one "
        "after another -- which is the 14 serial minutes this change exists to remove")
    assert result["learnings"], "a parallel round came back with no learnings"


@pytest.mark.asyncio
async def test_concurrency_is_bounded_by_the_run_not_by_the_level(unbudgeted):
    """The other half. A per-level semaphore would let each level open its own
    allowance, so the real ceiling would be concurrency x levels -- and underneath sits
    one embedding server that drops evidence when it is oversubscribed."""
    trace = _Trace()
    limit = 2
    await _run(trace, deep_research_concurrency=limit)

    assert trace.max_live <= limit, (
        f"{trace.max_live} nested researchers ran at once against a configured limit of "
        f"{limit} -- the semaphore is per recursion level, so every extra level "
        "multiplies the real load on the embedding server")


@pytest.mark.asyncio
async def test_one_failed_branch_does_not_take_its_siblings_down(unbudgeted):
    """A failed branch costs its own findings, never its siblings'.

    Targets the FIRST stage-2 branch's planned queries by name. (An earlier version of
    this test targeted a research goal, which never appears in a query string, so no
    branch ever failed and the test passed vacuously.)"""
    trace = _Trace()
    trace.fail_branch = "stage2 plan 1"
    _, result = await _run(trace)

    assert trace.failed > 0, (
        "non-vacuity: no nested researcher failed, so this test checks nothing")
    assert any("stage2 plan 2" in q for q in trace.completed), (
        f"the sibling branch's research did not complete after branch 1 failed "
        f"(completed: {trace.completed}) -- one failure took the round down")
    assert result["learnings"], (
        "one stage-2 branch failed and the whole run came back empty; the siblings' "
        "and stage 1's findings were discarded with it")


# --------------------------------------------------------------------------- steering
# Stage-2 planning used to see only its OWN branch's follow-up questions -- each proposed
# blind to the siblings -- so a branch could spend its round on ground a sibling had
# already answered. A filter after planning cannot fix that: the number of queries is
# fixed by then (see git history for the judge that tried). So the planner is steered
# instead, the way TreeResearchSkill.generate_child_questions is: shown what the WHOLE
# level found, asked for what is still missing, allowed to plan fewer or none.


@pytest.mark.asyncio
async def test_each_stage_two_branch_plans_against_every_siblings_findings(unbudgeted):
    """The defect: a branch planned from its own follow-ups alone. Every stage-2
    planning prompt must carry BOTH stage-1 branches -- the query and what it found."""
    trace = _Trace()
    await _run(trace)

    stage2 = [p for p in trace.plan_prompts if "ALREADY covered" in p]
    assert len(stage2) == BREADTH, (
        f"{len(stage2)} stage-2 planning calls carried the covered ground, expected one "
        f"per branch ({BREADTH}) -- a branch planned blind to what the run already found")
    for prompt in stage2:
        for query, learning in STAGE1:
            assert query in prompt and learning in prompt, (
                f"a stage-2 branch planned without seeing sibling {query!r} and what it "
                f"found ({learning!r}) -- it can re-research that ground and nothing "
                "would notice")


@pytest.mark.asyncio
async def test_the_top_level_plan_is_the_prompt_it_always_was(unbudgeted):
    """Nothing is covered before stage 1, so the first call must not change: a steered
    prompt there would redefine the run's opening for no information."""
    trace = _Trace()
    await _run(trace)

    top = [p for p in trace.plan_prompts if "ALREADY covered" not in p]
    assert len(top) == 1, f"expected exactly one unsteered (top-level) plan, got {len(top)}"
    assert (f"generate {BREADTH} unique search queries to research the topic thoroughly"
            in top[0]), "the top-level planning prompt changed"


@pytest.mark.asyncio
async def test_a_branch_the_planner_calls_answered_researches_nothing_more(unbudgeted):
    """What the steering buys that a filter could not: the round can SHRINK. A planner
    that judges the root answered plans no queries, and the run still returns stage 1."""
    trace = _Trace()
    trace.stage2_plan = ""
    _, result = await _run(trace)

    assert len(trace.built) == BREADTH, (
        f"{len(trace.built)} nested researchers ran after every stage-2 planner planned "
        f"nothing; only the {BREADTH} stage-1 queries should have")
    assert result["learnings"], "a round that planned nothing threw away stage 1 too"


@pytest.mark.asyncio
async def test_steering_costs_no_extra_llm_call(unbudgeted):
    """The coverage rides the planning call that already existed. One more call per
    level would be the judge's cost again under another name."""
    trace = _Trace()
    await _run(trace)

    assert len(trace.plan_prompts) == 1 + BREADTH, (
        f"{len(trace.plan_prompts)} planning calls for 1 top-level plan + {BREADTH} "
        "branches -- the steering added a call instead of riding the existing one")


# ------------------------------------------------------------------ planning in order
# Measured 2026-09-21, live: with every stage-2 branch planning concurrently against the
# same covered ground, all three picked the SAME gap (consumer idempotency). Each could
# see what stage 1 found but not what its siblings were about to plan. So branches now
# plan one after another, each shown the queries already planned, and research at once.


@pytest.mark.asyncio
async def test_a_later_branch_plans_around_what_earlier_branches_already_planned(unbudgeted):
    """The defect: sibling branches could not see each other's plans, so they converged.
    The first branch has nothing queued; the second must be shown the first's plan."""
    trace = _Trace()
    await _run(trace)

    stage2 = [p for p in trace.plan_prompts if "ALREADY covered" in p]
    assert len(stage2) == BREADTH, f"expected {BREADTH} stage-2 plans, got {len(stage2)}"
    assert "ALREADY PLANNED" not in stage2[0], (
        "the first branch was shown sibling plans that cannot exist yet")
    for planned in ("stage2 plan 1 a", "stage2 plan 1 b"):
        assert planned in stage2[1], (
            f"the second branch planned without seeing {planned!r}, which the first "
            "branch had already planned -- nothing stops both from picking the same gap")


@pytest.mark.asyncio
async def test_a_branch_whose_planning_fails_costs_only_that_branch(unbudgeted):
    """Planning moved out of the concurrent recursion into a loop, where an uncaught
    raise would end the whole round. It must cost that one branch, as it did before."""
    trace = _Trace()
    trace.fail_stage2_plan = 1
    _, result = await _run(trace)

    assert len(trace.built) == BREADTH + max(2, BREADTH // 2), (
        f"{len(trace.built)} nested researchers ran; expected stage 1 ({BREADTH}) plus "
        "the one branch whose planning succeeded")
    assert result["learnings"], "a single planning failure discarded the run"
