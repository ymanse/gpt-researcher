"""Round counters derived from gralph's OWN journal — not from the mutable store.

    python scripts/loop_audit.py
    -> rev_journal_s1=2 ... rev_journal_s5=0 bench_journal=3 grant_rev_s2=0 grant_bench=0

Why not `gralph.store`: the store is a plain JSON file in .gralph/, and an agent that hits
a round cap can simply reset the counter and keep going — measured on 2026-07-27, when a
session blocked by the s2 review cap rewound the cursor and set rev:s2 back to 0. The cap
exists to hand a decision to a HUMAN, so it must not be derived from something the blocked
party can edit.

journal.jsonl is append-only and framework-owned: every gate success is recorded with its
routing target, so the number of times a review actually routed back to impl (or the
benchmark routed back to a stage) is a fact about what the loop DID, not a counter someone
maintains. Tampering there means deleting the loop's own execution record, which also
destroys the session/gate history the audit reads.

Legitimate human grants (a reviewed decision to allow more rounds) live in
no_read/audit/grants.json, e.g. { "rev:s2": 3, "bench": 3 } — the number of journal-counted
rounds to forgive. The grant file is written by a human and surfaced by harness-audit, so a
raised cap is always visible instead of silently reset.
"""
from __future__ import annotations

import json

import hconf

JOURNAL = hconf.HARNESS / ".gralph" / "search-quality" / "journal.jsonl"
GRANTS = hconf.HARNESS / "no_read" / "audit" / "grants.json"


def counts() -> tuple[dict[int, int], int]:
    """(blocking review rounds per stage, benchmark refit rounds) from the journal."""
    rev = {n: 0 for n in range(1, 6)}
    bench = 0
    if not JOURNAL.exists():
        return rev, bench
    for ln in JOURNAL.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "command_succeeded":
            continue
        cmd, nxt = e.get("command", ""), e.get("next", "")
        for n in range(1, 6):
            # a review that routed back to impl == one blocking round actually spent
            if cmd == f"s{n}-review" and nxt == f"s{n}-impl":
                rev[n] += 1
        # a benchmark that routed anywhere except the audit == one refit round spent
        if cmd == "s6-benchmark" and nxt and nxt != "harness-audit":
            bench += 1
    return rev, bench


def grants() -> dict:
    if not GRANTS.exists():
        return {}
    try:
        return json.loads(GRANTS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    rev, bench = counts()
    g = grants()
    parts = [f"rev_journal_s{n}={rev[n]}" for n in range(1, 6)]
    parts.append(f"bench_journal={bench}")
    parts += [f"grant_rev_s{n}={int(g.get(f'rev:s{n}', 0))}" for n in range(1, 6)]
    parts.append(f"grant_bench={int(g.get('bench', 0))}")
    print(" ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
