"""Final benchmark runner — the ONLY writer of no_read/evidence/benchmark.json.

    python scripts/benchmark.py

Runs deep_tree_research for ALL golden queries (round-cached, interruption-safe), scores
each with score_report.py, aggregates (mean, int-rounded) and compares against the frozen
bench/baseline_firecrawl.json aggregate:

  pass rule: S1,S2,S4,S5,S6 aggregate STRICTLY ABOVE baseline AND S3 aggregate AT OR
  BELOW baseline. all_pass=1 only if every rule holds AND queries_scored == golden count.
  weakest_metric = the failing metric with the largest shortfall (ties -> lowest index),
  "-" when all pass.

Evidence: {bench_round, queries_scored, golden_count, per_metric:{Sk:{gptr,base,pass}},
all_pass, weakest_metric, errors[]}. Re-running with the round cache present re-scores
from cache — this script is its own law-10 regenerator.
"""
from __future__ import annotations

import json
import statistics

import hconf
import live_lib

HIGHER = ("S1", "S2", "S4", "S5", "S6")


def main() -> int:
    rnd = hconf.bench_round()
    live_lib.acquire_live_lock()
    ok = live_lib.recreate()
    health = live_lib.wait_health() if ok else 0

    goldens = hconf.load_golden()
    ev: dict = {"phase": "benchmark", "bench_round": rnd, "recreated": bool(ok),
                "health": health, "golden_count": len(goldens), "queries_scored": 0,
                "errors": [], "all_pass": 0, "weakest_metric": "-"}

    scores: list[dict] = []
    if health == 200:
        for g in goldens:
            run, err = live_lib.run_tree_cached(g, rnd)
            if not run:
                ev["errors"].append(f"{g['id']}: {err}")
                continue
            sc = live_lib.score(g["id"], run["report"], run["tree"],
                                f"no_read/bench_runs/round{rnd}/{g['id']}.scores.json")
            if not sc:
                ev["errors"].append(f"{g['id']}: score_report failed")
                continue
            scores.append(sc)
    ev["queries_scored"] = len(scores)

    if scores:
        try:
            base = json.loads(hconf.BASELINE.read_text(encoding="utf-8"))["aggregate"]
        except (OSError, KeyError, json.JSONDecodeError):
            base = {}
        per: dict = {}
        shortfalls: list[tuple[int, str]] = []
        for i in range(1, 7):
            k = f"S{i}"
            mine = int(statistics.mean(s.get(f"{k}_pct", 0) for s in scores))
            b = int(base.get(f"{k}_pct", 999 if k in HIGHER else -1))
            if k in HIGHER:
                passed = mine > b
                short = b - mine
            else:  # S3: lower is better
                passed = mine <= b
                short = mine - b
            per[k] = {"gptr": mine, "base": b, "pass": 1 if passed else 0}
            if not passed:
                shortfalls.append((short, k))
        ev["per_metric"] = per
        complete = ev["queries_scored"] == ev["golden_count"] and ev["golden_count"] >= 5
        if not shortfalls and complete:
            ev["all_pass"] = 1
        elif shortfalls:
            shortfalls.sort(key=lambda t: (-t[0], t[1]))
            ev["weakest_metric"] = shortfalls[0][1]

    hconf.write_json(hconf.EVID / "benchmark.json", ev)
    print(json.dumps(ev, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
