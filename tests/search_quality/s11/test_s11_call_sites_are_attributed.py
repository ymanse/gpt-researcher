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
    charged to each other's site and the retriever's ThreadPoolExecutor + asyncio.run
    hop still lands on the right site,
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
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpt_researcher.llm_provider.claude_agent import _subscription as sub


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


async def test_a_purpose_entered_in_a_worker_thread_is_seen_there_and_stays_there():
    """`smart_retriever` classifies from a sync `search()` called out of a running loop,
    so `_run_coro_blocking` hops through ThreadPoolExecutor + `asyncio.run`. The purpose
    is entered INSIDE the worker thread, around that hop: `asyncio.run` copies the
    calling THREAD's context into its task, so the tag must reach the call there — and
    must not follow the result back to the caller."""
    async def charge() -> dict:
        sub.note_agent_call()
        return sub.agent_calls_by_site()

    def worker() -> dict:
        with sub.agent_purpose("classify"):
            return asyncio.run(charge())

    with ThreadPoolExecutor(max_workers=1) as pool:
        seen_in_worker = pool.submit(worker).result(timeout=10)

    assert seen_in_worker.get("classify", 0) == 1, (
        "a purpose entered in the retriever's worker thread did not reach the call made "
        f"on that thread's own event loop: {seen_in_worker} — classify would be the one "
        "site that never gets attributed, and it is a site P1.1 exists to remove"
    )

    sub.note_agent_call()
    by_site = sub.agent_calls_by_site()
    assert by_site.get("classify", 0) == 1 and by_site.get("untagged", 0) == 1, (
        "the worker thread's purpose leaked back to the caller — the main-thread call "
        f"made after the hop must be 'untagged', got {by_site}"
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
