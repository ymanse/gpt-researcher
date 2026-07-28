"""Round counters derived from gralph's OWN journal — not from the mutable store.

    python scripts/loop_audit.py [--instance search-quality|dedup]
    -> rev_journal_s1=2 ... rev_journal_s9=0 bench_journal=3 off_journal_d1=0
       live_journal_d2=0 grant_rev_s1=0 ... grant_bench=0 grant_off_d1=0 grant_live_d2=0

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
raised cap is always visible instead of silently reset. Grants are SHARED across profiles
(one human decision log); the journals are per-instance, since each profile gets its own
.gralph/<instance>/ directory.
"""
from __future__ import annotations

import argparse
import json

import hconf

GRANTS = hconf.HARNESS / "no_read" / "audit" / "grants.json"
STAGES = range(1, 10)   # 1-5 search-quality lanes, 9 = dedup roll-up lane


def journal(instance: str):
    return hconf.HARNESS / ".gralph" / instance / "journal.jsonl"


def counts(instance: str) -> tuple[dict[int, int], int, int, int]:
    """(blocking review rounds per stage, benchmark refits, d1-offline refits,
    d2-live re-routes) from the journal."""
    rev = {n: 0 for n in STAGES}
    bench = off = live = 0
    jp = journal(instance)
    if not jp.exists():
        return rev, bench, off, live
    for ln in jp.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "command_succeeded":
            continue
        cmd, nxt = e.get("command", ""), e.get("next", "")
        for n in STAGES:
            # a review that routed back to impl == one blocking round actually spent
            if cmd == f"s{n}-review" and nxt == f"s{n}-impl":
                rev[n] += 1
        # a benchmark that routed anywhere except the audit == one refit round spent
        if cmd == "s6-benchmark" and nxt and nxt != "harness-audit":
            bench += 1
        # dedup: an offline measure that routed back to impl == one refit round spent
        if cmd == "d1-offline" and nxt and nxt != "d2-live":
            off += 1
        # dedup: a live confirm that did NOT reach the audit == one re-route spent
        if cmd == "d2-live" and nxt and nxt != "dedup-audit":
            live += 1
    return rev, bench, off, live


def grants() -> dict:
    if not GRANTS.exists():
        return {}
    try:
        return json.loads(GRANTS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="search-quality")
    a = ap.parse_args()

    rev, bench, off, live = counts(a.instance)
    g = grants()
    parts = [f"rev_journal_s{n}={rev[n]}" for n in STAGES]
    parts += [f"bench_journal={bench}", f"off_journal_d1={off}", f"live_journal_d2={live}"]
    parts += [f"grant_rev_s{n}={int(g.get(f'rev:s{n}', 0))}" for n in STAGES]
    parts += [f"grant_bench={int(g.get('bench', 0))}",
              f"grant_off_d1={int(g.get('off:d1', 0))}",
              f"grant_live_d2={int(g.get('live:d2', 0))}"]
    print(" ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
