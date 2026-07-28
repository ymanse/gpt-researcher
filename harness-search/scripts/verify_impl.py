"""IMPL (GREEN) verifier — run IN-GATE by impl_common.lua (and as agent preflight).

    python scripts/verify_impl.py --stage N

Recomputes, from disk truth (never from any agent claim):
  hash_match         — sha256 of every RED-recorded test file equals the RED record
  red_collected_ok   — stage collected count did not shrink below the RED count
  stage_*            — pytest re-run of tests/search_quality/sN (junit-derived)
  suite_*            — pytest re-run of the FULL cumulative tests/search_quality suite
  ruff_errors        — ruff check on files changed since base_sha (+ untracked py in scope)
  frozen_ok          — bench/golden + baseline hashes still match bench/manifest.sha256
  review_addressed_ok— if sN_review.json has blocking findings, every blocking id appears
                       in no_read/evidence/sN_impl_ack.json addressed_findings (agent-attested
                       engagement receipt; the real re-verdict is the NEXT review round)

Prints ONE machine line on stdout (the gate parses it) and writes
no_read/evidence/sN_impl.json as the audit trail. Idempotent; re-running it is also the
law-10 regenerator for impl evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import hconf


def ruff_changed_files(base_sha: str) -> int:
    files: set[str] = set()
    if base_sha:
        diff = hconf.git(hconf.REPO, "diff", "--name-only", f"{base_sha}..HEAD")
        files |= {f for f in diff.splitlines() if f.endswith(".py")}
    status = hconf.git(hconf.REPO, "status", "--porcelain")
    for ln in status.splitlines():
        f = ln[3:].strip().replace("\\", "/")
        if f.endswith(".py") and (f.startswith("gpt_researcher/") or f.startswith("tests/search_quality/")):
            files.add(f)
    files = {f for f in files if (hconf.REPO / f).exists()}
    if not files:
        return 0
    # ponytail: correctness rules only (E9 syntax, F pyflakes) — the fork carries ~35
    # pre-existing style violations (UP/BLE/I) per touched file; gating on those would
    # force unrelated churn. Widen the select if the build starts shipping style debt.
    r = subprocess.run(
        [str(hconf.VENV_PY), "-m", "ruff", "check", "--select", "E9,F",
         "--output-format", "json", *sorted(files)],
        capture_output=True, text=True, cwd=str(hconf.REPO), timeout=600,
    )
    try:
        return len(json.loads(r.stdout or "[]"))
    except json.JSONDecodeError:
        return 999  # ruff itself broke — fail closed


def check_frozen() -> int:
    r = subprocess.run([str(hconf.VENV_PY), str(hconf.HARNESS / "scripts" / "check_frozen.py")],
                       capture_output=True, text=True, timeout=120)
    return 1 if "frozen_ok=1" in (r.stdout or "") else 0


def review_addressed(stage: int) -> int:
    rev_path = hconf.EVID / f"s{stage}_review.json"
    if not rev_path.exists():
        return 1  # no review yet (first impl pass)
    try:
        rev = json.loads(rev_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    blocking = [f["id"] for f in rev.get("findings", []) if f.get("severity") == "blocking"]
    if not blocking:
        return 1
    ack_path = hconf.EVID / f"s{stage}_impl_ack.json"
    if not ack_path.exists():
        return 0
    try:
        ack = json.loads(ack_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    # Bind the ack to THIS review. Finding ids repeat across rounds (R1, R2, ... every
    # time), so an ack left over from the previous round would satisfy a brand-new
    # blocking finding by id collision alone — measured on the s2 round-3 stall.
    if ack.get("review_head_sha") != rev.get("head_sha"):
        return 0
    addressed = set(ack.get("addressed_findings", []))
    return 1 if all(b in addressed for b in blocking) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    # 1-5 = search-quality lanes; 9 = the dedup harness's roll-up lane (dedup.yaml).
    # Widened, never re-pointed: stages 1-5 resolve to exactly the same paths as before.
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 10))
    a = ap.parse_args()
    n = a.stage

    vals = {
        "red_evidence": 0, "hash_match": 0, "red_files": 0, "red_collected_ok": 0,
        "stage_collected": 0, "stage_failed": 0, "stage_errors": 1, "stage_skipped": 0,
        "suite_collected": 0, "suite_failed": 0, "suite_errors": 1, "suite_skipped": 0,
        "ruff_errors": 999, "frozen_ok": 0, "review_addressed_ok": 0,
    }
    reason = "-"

    red_path = hconf.EVID / f"s{n}_red.json"
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

        stage_counts = hconf.run_pytest(hconf.TESTS / f"s{n}", hconf.EVID / f"s{n}_impl_stage.xml")
        suite_counts = hconf.run_pytest(hconf.TESTS, hconf.EVID / f"s{n}_impl_suite.xml")
        for k, v in stage_counts.items():
            vals[f"stage_{k}"] = v
        for k, v in suite_counts.items():
            vals[f"suite_{k}"] = v
        vals["red_collected_ok"] = 1 if stage_counts["collected"] >= red.get("collected", 10**9) else 0
        vals["ruff_errors"] = ruff_changed_files(red.get("base_sha", ""))

    vals["frozen_ok"] = check_frozen()
    vals["review_addressed_ok"] = review_addressed(n)

    ok_all = (
        vals["red_evidence"] == 1 and vals["hash_match"] == 1 and vals["red_collected_ok"] == 1
        and vals["stage_collected"] >= 1 and vals["stage_failed"] == 0
        and vals["stage_errors"] == 0 and vals["stage_skipped"] == 0
        and vals["suite_collected"] >= vals["stage_collected"] and vals["suite_failed"] == 0
        and vals["suite_errors"] == 0 and vals["suite_skipped"] == 0
        and vals["ruff_errors"] == 0 and vals["frozen_ok"] == 1
        and vals["review_addressed_ok"] == 1
    )
    hconf.write_json(hconf.EVID / f"s{n}_impl.json",
                     {"stage": n, "phase": "impl", **vals, "ok": 1 if ok_all else 0})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} ok={1 if ok_all else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
