"""s13: hitting the subscription's session limit must not delete the research already done.

Measured 2026-09-10 05:12 on the container. A linear run armed 50, spent 2, and died
16 seconds later:

    [claude_agent] the CLI refused the request instead of answering:
    You've hit your session limit · resets 5:50am (UTC)
    ... Failed to get response from claude_agent API
    Research failed

Two failures wear that same last line, and only one of them is handled. A spent per-run
allowance raises AgentBudgetExceeded, and both the tree's expansion loop and the linear
path test `agent_budget_exhausted()` and finish with a shallower report. A provider-side
refusal is the same fact from the other direction — no further LLM call will succeed —
but nothing observes it: the internal counter is untouched, so the loops keep going and
the next call raises through the whole research run.

The asymmetry is the bug. It also cost a debugging session: the wrapper text is identical,
so a subscription limit reads as a code defect until someone opens the container log.
"""
from __future__ import annotations

import pytest

from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.utils.llm import is_llm_retryable_error

REFUSAL = ("[claude_agent] the CLI refused the request instead of answering: "
           "You've hit your session limit · resets 5:50am (UTC)")


@pytest.fixture(autouse=True)
def clean():
    sub.clear_provider_refusal()
    sub.begin_agent_run(1000)
    yield
    sub.clear_provider_refusal()
    sub.begin_agent_run(0)


def test_a_refusal_is_recorded_when_it_is_classified_as_non_retryable():
    """The one place every LLM failure already passes through, so nothing has to be
    remembered at each call site."""
    assert not sub.provider_refused()[0], "fixture precondition: no refusal recorded"

    assert is_llm_retryable_error(RuntimeError(REFUSAL)) is False, (
        "a refusal must stay non-retryable — backoff cannot refill a session limit")

    refused, reason = sub.provider_refused()
    assert refused, (
        "the refusal was classified but not RECORDED, so nothing downstream can see it. "
        "The loops that already stop on a spent allowance keep expanding into a provider "
        "that has stopped answering, and the next call kills the whole research run")
    assert "session limit" in reason and "5:50am" in reason, (
        f"the reason must carry what the CLI said, verbatim enough to act on: {reason!r}. "
        "Without the reset time the caller cannot tell a subscription limit from a bug")


def test_an_ordinary_error_records_nothing():
    """Only a refusal stops the run; a transient failure must stay transient."""
    assert is_llm_retryable_error(RuntimeError("connection reset by peer")) is True
    assert not sub.provider_refused()[0], (
        "a retryable error was recorded as a provider refusal, which would stop every "
        "later run in this process for a blip")


def test_research_stops_on_a_refusal_even_with_allowance_to_spare():
    """The contract the loops read. The budget is deliberately wide open: a refusal is
    not about this run's allowance, which is exactly why the existing check misses it."""
    assert not sub.research_should_stop(), "fixture precondition: nothing stops the run"

    is_llm_retryable_error(RuntimeError(REFUSAL))

    assert not sub.agent_budget_exhausted(), (
        "precondition: the per-run allowance is untouched — 1000 granted, ~0 spent. This "
        "is the state the 2026-09-10 failure was in when it died")
    assert sub.research_should_stop(), (
        "the run has an allowance to spare and a provider that refuses to answer, and "
        "nothing tells it to stop. That is the 16-second hard failure: 2 of 55 sessions "
        "spent, every gathered result discarded")


def test_clearing_lets_a_later_run_try_again():
    """A limit resets. The flag must not outlive it and disable research forever."""
    is_llm_retryable_error(RuntimeError(REFUSAL))
    assert sub.research_should_stop()

    sub.clear_provider_refusal()
    assert not sub.research_should_stop(), (
        "the refusal survived a clear, so once a container hits the limit once it never "
        "researches again until it restarts")


def test_arming_a_new_run_clears_a_stale_refusal():
    """begin_agent_run is the one thing every tool call does on entry, and a caller that
    starts a new run is asserting the limit may have reset."""
    is_llm_retryable_error(RuntimeError(REFUSAL))
    assert sub.research_should_stop()

    sub.begin_agent_run(50)
    assert not sub.research_should_stop(), (
        "a new tool call inherited the previous call's refusal and refused to research "
        "before trying anything. The limit resets on a clock this process cannot see, so "
        "the only honest move is to let the next run find out")
