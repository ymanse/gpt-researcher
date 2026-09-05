"""s11 / P0: a spent CLI session must say WHICH call site spent it.

Measured 2026-09-05 (harness-search/spec/p0-p1-slimming.md): one tree node costs 8-9
`claude` CLI sessions and 5-6 of them buy nothing — `choose_agent` once, smart
classification once inside `plan_research` and once more per sub-query,
`plan_research_outline` once. 20 nodes x 8 blows a 100-session cap, so the tree is cut
short by its own overhead. Today `agent_calls_spent()` reports only the total, which
says a run was expensive but never says which of those seven sites to slim — the P1
work (classify once per run, choose the agent once per run, bypass the planner) cannot
be measured before or after.

So this pins the attribution, not the total:

  - `agent_purpose(site)` tags every `note_agent_call()` made under it,
  - the tag rides a ContextVar, so concurrently-researched sibling nodes cannot be
    charged to each other's site — while the retriever's ThreadPoolExecutor hop lands on
    the right site ONLY once `_run_coro_blocking` copies the context at the submit site.
    `ThreadPoolExecutor.submit` alone starts the worker with an EMPTY context (measured
    2026-09-05: the var reads None there), so a plain ContextVar charges every production
    classify to 'untagged',
  - `agent_calls_this_run()` / `agent_calls_by_site()` are DELTAS re-baselined by
    `begin_agent_run()`, while `agent_calls_spent()` / `agent_budget_limit()` keep the
    cumulative, POOLING semantics that `tests/test_agent_budget_reserve.py` freezes —
    a per-run reset there was the 2026-08-06 defect that let four concurrent MCP
    streams share ONE 100-session allowance.

Hermetic: no CLI subprocess, no network, no LLM. The budget counters are charged
directly, which is the same seam `chat_model._run_query` uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
import threading

import pytest

from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.retrievers.smart.smart_retriever import _run_coro_blocking


# Big enough that no test here can trip AgentBudgetExceeded; the budget itself is
# somebody else's contract (tests/test_agent_budget_reserve.py).
ALLOWANCE = 1000


@pytest.fixture(autouse=True)
def armed_run(monkeypatch):
    """Start every test from a fresh baseline, and never touch the real ~/.claude.

    `begin_agent_run()` also runs `prune_cli_sessions()`, which deletes transcripts
    under the operator's real home directory. A unit test must not do housekeeping on
    a live machine, so the pruner is stubbed for the duration.
    """
    monkeypatch.setattr(sub, "prune_cli_sessions", lambda *a, **k: 0)
    sub.begin_agent_run(ALLOWANCE)
    yield
    # Disarm. The budget is process-wide and s11 sorts BEFORE s2..s9 and
    # tests/search_quality/test_*.py, so an allowance left armed here silently converts
    # every later tree run from unbounded to bounded (ceiling = spent + 1000, synthesis
    # reserve 150) and changes what those stages measure.
    sub.begin_agent_run(0)


def test_a_call_is_charged_to_the_purpose_that_wraps_it():
    """The basic contract: `agent_purpose` is a context manager, and the call made
    inside it is attributed to that site rather than merely counted."""
    with sub.agent_purpose("plan"):
        sub.note_agent_call()

    by_site = sub.agent_calls_by_site()
    assert by_site.get("plan", 0) == 1, (
        "a call made inside agent_purpose('plan') was not charged to 'plan' — only the "
        f"total moved, so there is still nothing to slim by. by_site={by_site}"
    )
    assert sub.agent_calls_this_run() == 1, (
        f"the run delta must see the same single call, got {sub.agent_calls_this_run()}"
    )


def test_every_declared_site_is_charged_under_its_own_name():
    """The site vocabulary is the reporting key `stats` and the linear tools expose, so
    a site must survive verbatim — folding one into another (or into 'untagged') hides
    exactly the overhead P1 is meant to remove."""
    sites = [s for s in sub.SITES if s != "untagged"]
    for site in sites:
        with sub.agent_purpose(site):
            sub.note_agent_call()

    by_site = sub.agent_calls_by_site()
    mischarged = {s: by_site.get(s, 0) for s in sites if by_site.get(s, 0) != 1}
    assert not mischarged, (
        f"declared sites did not receive their own call: {mischarged} (full breakdown "
        f"{by_site}) — a site that cannot be named cannot be measured or slimmed"
    )
    assert by_site.get("untagged", 0) == 0, (
        f"no tagged call may fall through to 'untagged': by_site={by_site}"
    )


def test_an_unattributed_call_lands_under_untagged():
    """Attribution must be total. A call made outside any purpose has to show up
    somewhere, or the per-site numbers silently stop adding up to the run's spend."""
    sub.note_agent_call()

    by_site = sub.agent_calls_by_site()
    assert by_site.get("untagged", 0) == 1, (
        f"a call made outside agent_purpose() vanished from the breakdown: {by_site}; "
        f"sum(by_site)={sum(by_site.values())} vs this_run={sub.agent_calls_this_run()}"
    )
    assert sum(by_site.values()) == sub.agent_calls_this_run(), (
        f"the breakdown must account for every call of the run: {by_site} sums to "
        f"{sum(by_site.values())}, this_run={sub.agent_calls_this_run()}"
    )


def test_beginning_a_run_rebaselines_the_delta_but_not_the_pooled_spend():
    """The new counters are per-run; the old ones stay cumulative.

    Resetting `agent_calls_spent()` is the 2026-08-06 defect that let four MCP streams
    armed two minutes apart share ONE 100-session allowance. The delta therefore has to
    be a separate baseline, not a reset of the counter the budget is enforced against.
    """
    for _ in range(3):
        with sub.agent_purpose("answer"):
            sub.note_agent_call()
    spent_before = sub.agent_calls_spent()
    assert sub.agent_calls_this_run() == 3, (
        f"three calls were charged, run delta reads {sub.agent_calls_this_run()}"
    )

    sub.begin_agent_run(ALLOWANCE)

    assert sub.agent_calls_this_run() == 0, (
        "arming a new run must re-baseline the delta to 0, otherwise this run's "
        "calls/node is measured against the previous run's spend, got "
        f"{sub.agent_calls_this_run()}"
    )
    assert sub.agent_calls_spent() == spent_before, (
        f"agent_calls_spent() must keep POOLING across runs — it dropped from "
        f"{spent_before} to {sub.agent_calls_spent()}, which is what let concurrent "
        "runs erase each other's spend"
    )
    assert sub.agent_budget_limit() == spent_before + ALLOWANCE, (
        "the ceiling must still rise by one allowance per run, got "
        f"{sub.agent_budget_limit()} instead of {spent_before + ALLOWANCE}"
    )

    with sub.agent_purpose("answer"):
        sub.note_agent_call()
    with sub.agent_purpose("answer"):
        sub.note_agent_call()
    assert sub.agent_calls_this_run() == 2, (
        "the delta must count only the calls made since arming, got "
        f"{sub.agent_calls_this_run()}"
    )
    assert sub.agent_calls_spent() == spent_before + 2, (
        f"the cumulative counter must keep rising through both runs: "
        f"{sub.agent_calls_spent()} instead of {spent_before + 2}"
    )


def test_beginning_a_run_rebaselines_the_per_site_breakdown_too():
    """`classify + choose_agent <= 2` is a PER-RUN gate. A breakdown that carried the
    previous run's sites forward would report the process, not the run."""
    with sub.agent_purpose("classify"):
        sub.note_agent_call()
    with sub.agent_purpose("choose_agent"):
        sub.note_agent_call()

    sub.begin_agent_run(ALLOWANCE)

    by_site = sub.agent_calls_by_site()
    assert by_site.get("classify", 0) == 0 and by_site.get("choose_agent", 0) == 0, (
        f"the previous run's sites survived begin_agent_run(): {by_site} — the gate "
        "'classify + choose_agent <= 2 per run' cannot be read off a process total"
    )

    with sub.agent_purpose("classify"):
        sub.note_agent_call()
    assert sub.agent_calls_by_site().get("classify", 0) == 1, (
        "the new run's own classify call must be visible after the re-baseline, got "
        f"{sub.agent_calls_by_site()}"
    )


async def test_concurrent_siblings_are_not_charged_to_each_others_site():
    """Sibling nodes research in parallel under `asyncio.gather`, which copies the
    context per Task. A purpose held in plain module state instead would let whichever
    sibling entered last own BOTH calls, and the breakdown would blame one node for
    another node's spend."""
    a_entered = asyncio.Event()
    b_entered = asyncio.Event()

    async def sibling(site: str, mine: asyncio.Event, theirs: asyncio.Event) -> None:
        with sub.agent_purpose(site):
            mine.set()
            # force the interleave: both siblings are inside their purpose at once
            await asyncio.wait_for(theirs.wait(), 5)
            sub.note_agent_call()

    await asyncio.gather(
        sibling("answer", a_entered, b_entered),
        sibling("children", b_entered, a_entered),
    )

    by_site = sub.agent_calls_by_site()
    assert by_site.get("answer", 0) == 1 and by_site.get("children", 0) == 1, (
        "two coroutines held different purposes at the same time and their calls were "
        f"not kept apart: {by_site} — with a shared (non-ContextVar) purpose the last "
        "sibling to enter takes the whole charge"
    )
    assert by_site.get("untagged", 0) == 0, (
        f"neither sibling's tag may be lost across the await: {by_site}"
    )


async def test_a_purpose_survives_the_retrievers_real_thread_hop():
    """The topology production actually has, driven through the REAL `_run_coro_blocking`.

    `SmartRetriever.search()` is synchronous but is called from a thread that ALREADY
    owns a running loop — the MCP server, and the direct `retriever_instance.search(...)`
    calls at researcher.py:656 and :1001. So `_run_coro_blocking` takes its second
    branch, `pool.submit(lambda: asyncio.run(coro))`, and **ThreadPoolExecutor.submit
    does not copy contextvars**: the spawned thread starts with an EMPTY context and a
    purpose entered by the caller is silently lost (measured 2026-09-05 — the var reads
    None there).

    The purpose is therefore entered HERE, on the calling thread, around the whole
    blocking hop. That is the only place production can enter it (`search()` is what the
    caller holds) and the only place it belongs — around the logical operation. A test
    that instead enters the purpose inside its own worker thread certifies nothing: a
    plain ContextVar passes it while production charges everything to 'untagged'.
    """
    caller_thread = threading.get_ident()
    seen: dict = {}

    async def charge() -> None:
        seen["thread"] = threading.get_ident()
        sub.note_agent_call()

    with sub.agent_purpose("classify"):
        _run_coro_blocking(charge())

    assert seen["thread"] != caller_thread, (
        "this test did not cross the hop it exists for: _run_coro_blocking ran the "
        f"coroutine on the calling thread ({caller_thread}) instead of spawning one, so "
        "the branch the MCP path takes went untested and the assertion below would pass "
        "vacuously"
    )

    by_site = sub.agent_calls_by_site()
    assert by_site.get("classify", 0) == 1, (
        "a call made under agent_purpose('classify') across the retriever's thread hop "
        f"was charged elsewhere: by_site={by_site}. ThreadPoolExecutor.submit does not "
        "copy contextvars, so the spawned thread starts with an empty context and EVERY "
        "production classify lands under 'untagged' — which makes the gate "
        "'classify + choose_agent <= 2 per run' pass vacuously by reading 0. The fix "
        "belongs at the submit site in smart_retriever._run_coro_blocking: "
        "ctx = contextvars.copy_context(); pool.submit(lambda: ctx.run(asyncio.run, coro))"
    )
    assert by_site.get("untagged", 0) == 0, (
        f"the tag was dropped somewhere across the hop: by_site={by_site}"
    )


def test_the_branch_that_does_not_hop_keeps_the_purpose_too():
    """`_run_coro_blocking`'s other branch: no loop on this thread, so it drives
    `asyncio.run` in place, which copies the CALLING thread's context into its task.

    This half is already correct today — it is pinned because the fix for the hopping
    branch edits this same function, and a `copy_context()` bolted on carelessly (e.g.
    entering the copied context around the wrong call, or reusing one context for both
    branches) is exactly how it would regress.
    """
    caller_thread = threading.get_ident()
    seen: dict = {}

    async def charge() -> None:
        seen["thread"] = threading.get_ident()
        sub.note_agent_call()

    with sub.agent_purpose("plan"):
        _run_coro_blocking(charge())

    assert seen["thread"] == caller_thread, (
        "a loop was running on this thread after all, so this test drove the hopping "
        "branch and duplicates the previous test instead of covering the other one"
    )
    by_site = sub.agent_calls_by_site()
    assert by_site.get("plan", 0) == 1 and by_site.get("untagged", 0) == 0, (
        "the no-running-loop branch of _run_coro_blocking lost the purpose: "
        f"by_site={by_site} — asyncio.run copies the calling thread's context, so this "
        "branch must attribute correctly both before and after the submit-site fix"
    )


@contextlib.contextmanager
def _never_armed_copy():
    """Yield a fresh import of the counter module — i.e. a process that imported
    `gpt_researcher` and never called `begin_agent_run()`.

    That state cannot be faked from here: the baseline globals are the implementation's
    business, not the spec's, and `begin_agent_run(0)` would not reproduce it either —
    disarming is still an arming, while rule 4c is about the baseline BEFORE the first
    one. A second module object IS process start, and it is isolated from the module the
    autouse fixture armed, so this cannot disturb the other tests. Restored in `finally`
    so a failure cannot leave a hole in `sys.modules`.
    """
    name = sub.__name__
    pkg, attr = name.rsplit(".", 1)
    original = sys.modules[name]
    del sys.modules[name]
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules[name] = original
        setattr(sys.modules[pkg], attr, original)


def test_an_unarmed_process_reports_its_whole_spend_as_this_run():
    """Rule 4c: before any `begin_agent_run()`, the baseline is process start.

    This is not a corner case — it is how the measurement is actually taken. The
    search-quality harness imports `gpt_researcher` directly and never arms; only the
    MCP tools call `begin_agent_run()`. A delta that stays 0 (or a breakdown that stays
    empty) until somebody arms would report zero for every harness run, and the calls/node
    and `classify + choose_agent` gates would be graded on a number nothing ever moved.
    """
    with _never_armed_copy() as fresh:
        assert fresh.agent_budget_limit() == 0 and fresh.agent_calls_spent() == 0, (
            "precondition failed — this copy is not a never-armed process start: "
            f"limit={fresh.agent_budget_limit()} spent={fresh.agent_calls_spent()}"
        )

        with fresh.agent_purpose("plan"):
            fresh.note_agent_call()
        fresh.note_agent_call()

        assert fresh.agent_calls_this_run() == fresh.agent_calls_spent() == 2, (
            "with no begin_agent_run() in this process the run delta must equal the "
            f"pooled spend: this_run={fresh.agent_calls_this_run()} "
            f"spent={fresh.agent_calls_spent()} — a delta that only starts counting once "
            "somebody arms reads 0 for the whole harness, which never arms"
        )

        by_site = fresh.agent_calls_by_site()
        assert sum(by_site.values()) == fresh.agent_calls_spent(), (
            f"the unarmed breakdown must cover every call so far: {by_site} sums to "
            f"{sum(by_site.values())} against {fresh.agent_calls_spent()} calls spent"
        )
        assert by_site.get("plan", 0) == 1 and by_site.get("untagged", 0) == 1, (
            f"the unarmed breakdown lost the tagged/untagged split: {by_site}"
        )


def test_leaving_a_purpose_restores_the_one_it_was_nested_in():
    """Node research enters `answer`, and the retriever it drives enters `classify`
    underneath it. If leaving the inner purpose cleared the tag instead of restoring the
    outer one, everything after a node's first retriever call would be charged to
    'untagged' and the node's own cost would disappear."""
    with sub.agent_purpose("answer"):
        with sub.agent_purpose("classify"):
            sub.note_agent_call()
        sub.note_agent_call()

        try:
            with sub.agent_purpose("children"):
                raise RuntimeError("an LLM call inside a purpose failed")
        except RuntimeError:
            pass
        sub.note_agent_call()

    sub.note_agent_call()

    by_site = sub.agent_calls_by_site()
    assert by_site.get("classify", 0) == 1, (
        f"the nested purpose must take exactly its own call: {by_site}"
    )
    assert by_site.get("answer", 0) == 2, (
        "leaving a nested purpose must restore the enclosing one — both post-nesting "
        "calls belong to 'answer', including the one after the inner block raised: "
        f"{by_site}"
    )
    assert by_site.get("untagged", 0) == 1, (
        f"only the call made after the outermost purpose exited is untagged: {by_site}"
    )
