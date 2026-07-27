"""Live measure runner — the ONLY writer of no_read/evidence/sN_measure.json.

    python scripts/measure.py --stage N     (N in 1..5)

Common preamble: acquire the live lock, force-recreate the container (bind-mounted source
must be live), wait /health 200, stamp bench_round from the gralph store.

  s1: for EVERY golden query (5): docker-cp + docker-exec container_probe_s1.py, parse the
      SQ_PROBE line -> scraped_pages / context_chars / retriever_errors per query. Fields:
      queries_run, scraped_pages_min, retriever_errors_total, context_chars_median.
  s2: for the 2 measure_pair goldens: deep_tree_research (round-cached) + score_report.
      Fields: queries_run, S1_min_pct, uncited_ids_total.
  s3: same runs. Fields: queries_run, traps_hit_total.
  s4: same runs. Fields: queries_run, S4_min_pct, pruned_count_total, s5_improved
      (mean S5_pct over measured queries > baseline mean over the same queries).
  s5: same runs. Fields: queries_run, contradictions_total, unsupported_claims_total.

Per-query raw artifacts cache under no_read/bench_runs/round{R}/ — re-running skips
completed queries (interruption-safe) and re-scores from cache, which makes this script
its own law-10 regenerator. Failed queries are recorded in "errors" and leave the
threshold fields at fail-closed defaults.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess

import code_fp
import hconf
import live_lib


def probe_s1(golden: dict) -> dict | None:
    src = hconf.HARNESS / "scripts" / "container_probe_s1.py"
    subprocess.run(["docker", "cp", str(src), f"{hconf.CONTAINER}:/tmp/probe_s1.py"],
                   capture_output=True, timeout=120)
    r = subprocess.run(
        ["docker", "exec", hconf.CONTAINER, "python", "/tmp/probe_s1.py",
         golden["id"], golden["query"]],
        capture_output=True, text=True, errors="replace", timeout=1800,
    )
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("SQ_PROBE "):
            try:
                return json.loads(ln[len("SQ_PROBE "):])
            except json.JSONDecodeError:
                return None
    return None


def tree_scores(goldens: list[dict], rnd: int, errors: list[str]) -> list[dict]:
    out = []
    for g in goldens:
        run, err = live_lib.run_tree_cached(g, rnd)
        if not run:
            errors.append(f"{g['id']}: {err}")
            continue
        sc = live_lib.score(g["id"], run["report"], run["tree"],
                            f"no_read/bench_runs/round{rnd}/{g['id']}.scores.json")
        if not sc:
            errors.append(f"{g['id']}: score_report failed")
            continue
        out.append(sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 6))
    a = ap.parse_args()
    n = a.stage
    rnd = hconf.bench_round()

    live_lib.acquire_live_lock()
    ok = live_lib.recreate()
    health = live_lib.wait_health() if ok else 0
    ev: dict = {"stage": n, "phase": "measure", "bench_round": rnd,
                "recreated": bool(ok), "health": health, "queries_run": 0, "errors": [],
                # the implementation these numbers describe; the gate recomputes it and
                # rejects a mismatch, so evidence can never outlive the code it measured
                "code_fp": code_fp.fingerprint()}

    if health == 200:
        if n == 1:
            probes = []
            for g in hconf.load_golden():
                p = probe_s1(g)
                if p:
                    probes.append(p)
                else:
                    ev["errors"].append(f"{g['id']}: probe produced no SQ_PROBE line")
            ev["queries_run"] = len(probes)
            ev["probes"] = probes
            if probes:
                ev["scraped_pages_min"] = min(p["scraped_pages"] for p in probes)
                ev["retriever_errors_total"] = sum(p["retriever_errors"] for p in probes)
                ev["context_chars_median"] = int(statistics.median(p["context_chars"] for p in probes))
                # law 4: retriever_errors counts records carrying the unrecovered marker,
                # so a marker nobody emits would make that 0 meaningless. Bind it.
                ev["marker_wired"] = 1 if all(p.get("marker_wired") for p in probes) else 0
            else:
                ev["scraped_pages_min"] = 0
                ev["retriever_errors_total"] = 999
                ev["context_chars_median"] = 0
                ev["marker_wired"] = 0
        else:
            pair = hconf.measure_pair()
            scores = tree_scores(pair, rnd, ev["errors"])
            ev["queries_run"] = len(scores)
            ev["scores"] = [{k: s.get(k) for k in
                            ("qid", "S1_pct", "S2_pct", "S3_pct", "S4_pct", "S5_pct", "S6_pct",
                             "uncited_ids", "traps_hit", "contradictions", "unsupported_claims")}
                           for s in scores]
            if scores:
                if n == 2:
                    ev["S1_min_pct"] = min(s.get("S1_pct", 0) for s in scores)
                    # uncited_ids is the list of unresolved [id]s (bench/score_report.py),
                    # not a count -- sum its length, fail-closed to 999 if a score is missing it
                    ev["uncited_ids_total"] = sum(
                        len(s["uncited_ids"]) if isinstance(s.get("uncited_ids"), list) else 999
                        for s in scores
                    )
                elif n == 3:
                    ev["traps_hit_total"] = sum(s.get("traps_hit", 999) for s in scores)
                elif n == 4:
                    ev["S4_min_pct"] = min(s.get("S4_pct", 0) for s in scores)
                    pruned = 0
                    for g in pair:
                        tp = hconf.BENCH_RUNS / f"round{rnd}" / f"{g['id']}.tree.json"
                        try:
                            pruned += int(json.loads(tp.read_text(encoding="utf-8"))
                                          ["meta"]["pruned_count"])
                        except (OSError, KeyError, ValueError, json.JSONDecodeError):
                            pass
                    ev["pruned_count_total"] = pruned
                    try:
                        base = json.loads(hconf.BASELINE.read_text(encoding="utf-8"))["queries"]
                        base_mean = statistics.mean(base[s["qid"]]["S5_pct"] for s in scores)
                        mine_mean = statistics.mean(s.get("S5_pct", 0) for s in scores)
                        ev["s5_improved"] = 1 if mine_mean > base_mean else 0
                        ev["S5_mean_pct"] = int(mine_mean)
                        ev["S5_baseline_mean_pct"] = int(base_mean)
                    except (OSError, KeyError, ValueError, json.JSONDecodeError):
                        ev["s5_improved"] = 0
                elif n == 5:
                    ev["contradictions_total"] = sum(s.get("contradictions", 999) for s in scores)
                    ev["unsupported_claims_total"] = sum(s.get("unsupported_claims", 999) for s in scores)

    hconf.write_json(hconf.EVID / f"s{n}_measure.json", ev)
    print(json.dumps(ev, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
