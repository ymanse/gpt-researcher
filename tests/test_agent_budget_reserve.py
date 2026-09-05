"""The synthesis reserve must stay proportional to ONE run's allowance.

Regression for 2026-08-16: allowances pool across begin_agent_run() calls, so the
reserve — then a fraction of the POOLED ceiling — grew every run and ate the next
run's expansion headroom (47/35/21/11 calls over four 100-call runs).

Run: python -m pytest tests/test_agent_budget_reserve.py
"""
from gpt_researcher.llm_provider.claude_agent._subscription import (
    agent_budget_exhausted,
    agent_synthesis_reserve,
    begin_agent_run,
    note_agent_call,
)


def test_reserve_does_not_grow_as_allowances_pool():
    for run in range(4):
        begin_agent_run(100)
        assert agent_synthesis_reserve() == 15, f"run {run + 1} reserve drifted"
        # spend this run's whole allowance so the next one arms on a raised ceiling
        for _ in range(100):
            note_agent_call()


def test_expansion_headroom_is_one_allowance_minus_reserve_every_run():
    for run in range(4):
        begin_agent_run(100)
        headroom = 0
        while not agent_budget_exhausted(reserve=agent_synthesis_reserve()):
            note_agent_call()
            headroom += 1
        assert headroom == 85, f"run {run + 1} got {headroom} calls, expected 85"
        for _ in range(15):  # the synthesis the reserve was held back for
            note_agent_call()


def test_old_ceiling_denominated_reserve_is_what_starved_run_four():
    """Pins the defect itself: the pre-fix formula, replayed, reproduces 47/35/21/11.

    Can't diff against HEAD — the whole budget module is still uncommitted — so the
    old rule (reserve = 15% of the POOLED ceiling) is spelled out here instead.
    """
    # measured agent_calls_spent at each run's arming, 2026-08-16
    spent_at_arming = [258, 338, 430, 495]
    headroom_per_run = []
    for spent in spent_at_arming:
        ceiling = spent + 100
        old_reserve = max(5, ceiling * 15 // 100)  # the bug: ceiling, not allowance
        headroom_per_run.append(max(0, (ceiling - old_reserve) - spent))
    assert headroom_per_run == [47, 35, 21, 11]

    # the fix keeps every run at the same 85, no matter how deep the pool is
    assert [100 - max(5, 100 * 15 // 100)] * 4 == [85, 85, 85, 85]


def test_unbounded_run_reserves_nothing():
    begin_agent_run(0)
    assert agent_synthesis_reserve() == 0
    assert agent_budget_exhausted() is False


def test_pruner_deletes_stale_transcripts_but_spares_live_ones(tmp_path, monkeypatch):
    """The pruner must never delete a transcript a running session is still writing."""
    import os as _os
    import time as _time

    from gpt_researcher.llm_provider.claude_agent import _subscription

    projects = tmp_path / ".claude" / "projects" / "-app"
    projects.mkdir(parents=True)
    stale, live = projects / "old.jsonl", projects / "running.jsonl"
    stale.write_text("{}"), live.write_text("{}")
    old = _time.time() - 5 * 86400
    _os.utime(stale, (old, old))

    monkeypatch.setattr(_os.path, "expanduser", lambda p: str(tmp_path))

    assert _subscription.prune_cli_sessions(retention_days=2) == 1
    assert not stale.exists()
    assert live.exists(), "a transcript younger than the window must survive"

    # retention_days<=0 is the opt-out, not "delete everything"
    _os.utime(live, (old, old))
    assert _subscription.prune_cli_sessions(retention_days=0) == 0
    assert live.exists()
