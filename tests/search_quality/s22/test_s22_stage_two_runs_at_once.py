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
  - a failed branch costs its own findings, never its siblings'.

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


def _answer_blob(prompt: str) -> str:
    """Learnings plus the follow-up belonging to whichever branch is answering."""
    question = FOLLOW_UPS[1] if "deduplicate" in prompt else FOLLOW_UPS[0]
    return (f"Learning [{DOC_URL}]: The outbox table grows without bound unless a job "
            f"trims it.\nQuestion: {question}\n")


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

    async def llm(self, *args, **kwargs):
        """One fake for every call site, told apart by what the prompt asks for."""
        messages = kwargs.get("messages") or []
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "key learnings" in text:
            return _answer_blob(text)
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
                    raise RuntimeError("branch exploded")
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
    """`asyncio.gather` without return_exceptions raises on the first failure and
    discards the rest -- including branches that had already finished."""
    trace = _Trace()
    trace.fail_branch = "retention practice"
    _, result = await _run(trace)

    assert result["learnings"], (
        "one stage-2 branch raised and the whole run came back empty; the siblings' "
        "findings were discarded with it")
