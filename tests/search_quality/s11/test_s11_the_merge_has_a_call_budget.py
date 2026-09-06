"""s11: the roll-up's de-duplication must not eat the run's whole allowance.

Measured 2026-09-05 across four live runs of the same 3-node query. The research got
cheap and the SYNTHESIS did not move, so the roll-up became the dominant cost:

    agent_calls_by_site: {classify 1, choose_agent 1, answer 3, children 1, merge 12}

12 of 18 sessions — 67% — went to the claim-equivalence judge, and bought a 7% shorter
report (15,442 -> 14,343 chars, 3 of 59 claim units merged). It scales with CLAIM COUNT,
not with nodes: the harness recorded 22 screens on a 167-unit tree, so a full 20-node run
is far worse than what these numbers show.

The operator is not the problem and is not touched here — the harness already established
that strict equivalence can only remove a few percent, and s9 freezes what it does remove.
What is missing is a CEILING. So:

  * merge spends at most a bounded number of calls on a run whose budget is armed,
  * it spends them on the strongest candidates first (groups are already grown around the
    closest pair, and verdict pairs are already ordered by score),
  * and running out is SAFE: merging only ever deletes text, so an unscreened unit simply
    survives into the report. Stopping early costs redundancy, never a fact.

Unbounded runs (the harness imports gpt_researcher directly and never arms) keep today's
behaviour exactly, which is what leaves s9's contract untouched.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.tree_research as tr
from gpt_researcher.llm_provider.claude_agent import _subscription as sub


# Enough near-identical units that ungated merging would screen many groups.
UNITS = [f"The outbox relay can deliver message {i} more than once." for i in range(24)]


@pytest.fixture(autouse=True)
def disarmed():
    """Never leak an allowance: s11 sorts before every other stage."""
    yield
    sub.begin_agent_run(0)


def _skill():
    parent = SimpleNamespace(
        query="outbox duplicate delivery",
        cfg=SimpleNamespace(strategic_llm_provider="fake", strategic_llm_model="m",
                            fast_llm_provider="fake", fast_llm_model="f",
                            smart_llm_provider="fake", smart_llm_model="s",
                            llm_kwargs={}, config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set())
    return tr.TreeResearchSkill(parent)


async def _run_merge(skill, monkeypatch):
    """Drive _merge_claim_units with every model seam counted, none of them real."""
    calls = {"screen": 0, "verdict": 0}

    async def _covered(texts):
        calls["screen"] += 1
        sub.note_agent_call()
        # everything after the first says nothing of its own -> maximal merge pressure
        return {i: i > 0 for i in range(len(texts))}

    async def _droppable(a, b):
        calls["verdict"] += 1
        sub.note_agent_call()
        return 1

    monkeypatch.setattr(skill, "_covered", _covered)
    monkeypatch.setattr(skill, "_droppable", _droppable)

    async def _vectors(texts):
        # one tight cluster: every unit is a near-duplicate of every other
        return [[1.0, 0.001 * i] for i in range(len(texts))], tr._cosine

    monkeypatch.setattr(skill, "_unit_vectors", _vectors)
    vecs, sim = await _vectors(UNITS)
    await skill._merge_claim_units(list(UNITS), vecs, sim)
    return calls


@pytest.mark.asyncio
async def test_a_bounded_run_caps_what_the_merge_may_spend(monkeypatch):
    """The ceiling this stage is about."""
    sub.begin_agent_run(40)
    skill = _skill()
    before = sub.agent_calls_spent()

    calls = await _run_merge(skill, monkeypatch)
    spent = sub.agent_calls_spent() - before
    cap = tr.merge_call_budget()

    assert cap > 0, "a bounded run must derive a merge ceiling"
    assert spent <= cap, (
        f"the merge spent {spent} CLI sessions against a ceiling of {cap}. Measured live, "
        f"it took 12 of a run's 18 sessions to shorten the report by 7%, and it scales "
        f"with claim count — on a full tree it is the run"
    )
    assert calls["screen"] >= 1, (
        f"non-vacuity: the merge must still do its job, screens={calls['screen']}")


@pytest.mark.asyncio
async def test_the_ceiling_is_reported_so_a_truncated_merge_is_visible(monkeypatch):
    """A merge that stopped early must say so, or the report looks fully de-duplicated."""
    sub.begin_agent_run(40)
    skill = _skill()
    await _run_merge(skill, monkeypatch)

    stats = skill._merge_calls
    assert "call_budget" in stats and "budget_exhausted" in stats, (
        f"_merge_calls carries {sorted(stats)} — without the ceiling and whether it was "
        f"reached, a run that de-duplicated half its claims and one that de-duplicated "
        f"all of them report the same numbers")


@pytest.mark.asyncio
async def test_an_unbounded_run_is_not_capped(monkeypatch):
    """The harness imports gpt_researcher directly and never arms; s9's contract is its."""
    sub.begin_agent_run(0)
    skill = _skill()

    calls = await _run_merge(skill, monkeypatch)

    assert tr.merge_call_budget() == 0, (
        "an unbounded run must derive no merge ceiling, or the library caller silently "
        "gets a different de-duplication than the one s9 froze")
    assert calls["screen"] >= 1 and skill._merge_calls["budget_exhausted"] is False, (
        f"an unbounded merge reported itself truncated: {skill._merge_calls}")


@pytest.mark.asyncio
async def test_the_ceiling_is_denominated_in_this_runs_allowance_not_the_pool():
    """The same defect the synthesis reserve already had, caught live on the first
    tiered run.

    Allowances POOL: `agent_budget_limit()` is the sum of every allowance the process has
    granted, so it climbs for the life of the container (observed: 651 before a restart).
    A merge ceiling denominated in it grows without bound — a `standard` call that asked
    for 25 was measured getting a merge budget of 7, because a `light` call had run first
    and the pooled ceiling was 30.

    THIS run's allowance is the only denominator that means anything: the ceiling has to
    say "a quarter of what I asked for", not "a quarter of everything anyone asked for".
    """
    sub.begin_agent_run(12)          # a light call
    for _ in range(5):
        sub.note_agent_call()
    sub.begin_agent_run(25)          # then a standard one, pooling on top of it

    assert sub.agent_budget_limit() > 25, (
        "fixture precondition: the ceiling must have pooled above this run's allowance")
    assert tr.merge_call_budget() == max(tr.MERGE_CALL_FLOOR,
                                         25 * tr.MERGE_CALL_PCT // 100), (
        f"the merge ceiling is {tr.merge_call_budget()} — derived from the pooled "
        f"ceiling {sub.agent_budget_limit()} rather than from this run's allowance of 25. "
        f"It therefore grows every time any call arms a budget, which is exactly the bug "
        f"agent_synthesis_reserve was already fixed for")
