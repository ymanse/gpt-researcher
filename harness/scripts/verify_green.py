"""GREEN verifier — run IN-GATE by green_common.lua (and as agent preflight).

    python scripts/verify_green.py --stage N

Recomputes, from disk truth (never from any agent claim):
  hash_match       — sha256 of every RED-recorded test file equals the RED record
  red_collected_ok — stage collected count did not shrink below the RED count
  stage_*          — pytest re-run of tests/tier_a/stageN (junit-derived)
  suite_*          — pytest re-run of the FULL cumulative tests/tier_a suite (regression)

Prints ONE machine line on stdout (the gate parses it) and writes
no_read/evidence/stageN_green.json as the audit trail. Idempotent; re-running it is
also the law-10 regenerator for green evidence.
"""
from __future__ import annotations

import argparse
import json

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 7))
    a = ap.parse_args()
    n = a.stage

    vals = {
        "red_evidence": 0, "hash_match": 0, "red_files": 0, "red_collected_ok": 0,
        "stage_collected": 0, "stage_failed": 0, "stage_errors": 1, "stage_skipped": 0,
        "suite_collected": 0, "suite_failed": 0, "suite_errors": 1, "suite_skipped": 0,
    }
    reason = "-"

    red_path = hconf.EVID / f"stage{n}_red.json"
    red = None
    if red_path.exists():
        try:
            red = json.loads(red_path.read_text(encoding="utf-8"))
            vals["red_evidence"] = 1
        except json.JSONDecodeError:
            reason = "red_evidence_unparseable"
    else:
        reason = "red_evidence_missing"

    if red:
        files = red.get("test_files", {})
        vals["red_files"] = len(files)
        ok = bool(files)
        for rel, h in files.items():
            p = hconf.REPO / rel
            if not p.exists() or hconf.sha256(p) != h:
                ok = False
                reason = f"modified_or_missing:{rel}"
        vals["hash_match"] = 1 if ok else 0

        stage_counts = hconf.run_pytest(hconf.TESTS / f"stage{n}", hconf.EVID / f"stage{n}_green_stage.xml")
        suite_counts = hconf.run_pytest(hconf.TESTS, hconf.EVID / f"stage{n}_green_suite.xml")
        for k, v in stage_counts.items():
            vals[f"stage_{k}"] = v
        for k, v in suite_counts.items():
            vals[f"suite_{k}"] = v
        vals["red_collected_ok"] = 1 if stage_counts["collected"] >= red.get("collected", 10**9) else 0

    ok_all = (
        vals["red_evidence"] == 1 and vals["hash_match"] == 1 and vals["red_collected_ok"] == 1
        and vals["stage_collected"] >= 1 and vals["stage_failed"] == 0
        and vals["stage_errors"] == 0 and vals["stage_skipped"] == 0
        and vals["suite_collected"] >= vals["stage_collected"] and vals["suite_failed"] == 0
        and vals["suite_errors"] == 0 and vals["suite_skipped"] == 0
    )
    hconf.write_json(hconf.EVID / f"stage{n}_green.json",
                     {"stage": n, "phase": "green", **vals, "ok": 1 if ok_all else 0})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} ok={1 if ok_all else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
