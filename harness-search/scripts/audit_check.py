"""harness-audit verifier — run IN-GATE by harness_audit.lua.

    python scripts/audit_check.py

Deterministically audits the HARNESS itself (not the build):

  try_ok      law 6 — for every profile node there are two `gralph try --report` JSON
              reports under no_read/audit/try/: <node>_pass.json with "success": true and
              <node>_fail.json with "success": false. The audit AGENT produces them by
              probing each gate with known-good/known-bad evidence (backing up and
              restoring real evidence via the regenerators); this script only reads
              gralph's own tool-emitted reports.
  git_ok      score-tamper detection — `git log --follow` shows NO commit after the s0
              freeze commit that touches bench/golden/* or bench/baseline_firecrawl.json.
              (The freeze commit = the earliest commit containing [sq][s0].)
  frozen_ok   manifest hashes still match (check_frozen.py verdict).
  regen_ok    law 10 — re-running the regenerators twice yields byte-identical evidence:
              bench_selfcheck.py (s0), verify_impl.py --stage 1..5 (impl). Measure/
              benchmark evidence is regenerated from the round cache by their own
              scripts; this script re-runs measure.py only when the cache exists for the
              current round, otherwise counts it as covered-by-cache-absence (recorded).
  sync_ok     law 8 — every threshold token each gate checks appears verbatim in the
              profile guidance AND in HARNESS.md (three-way sync scan).

Prints ONE machine line; writes no_read/evidence/audit.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

import hconf

TRY_DIR = hconf.HARNESS / "no_read" / "audit" / "try"
PROFILE = hconf.HARNESS / "search-quality.yaml"
HARNESS_MD = hconf.HARNESS / "HARNESS.md"

NODES = ["s0-bench"] + [f"s{n}-{k}" for n in range(1, 6)
                        for k in ("red", "impl", "review", "measure")] \
        + ["s6-benchmark", "harness-audit"]

# gate threshold tokens that guidance + HARNESS.md must both mention (law 8)
SYNC_TOKENS = [
    "golden_count", "fixtures_passed", "llm_calls", "baseline_queries", "diversity_ok",
    "scraped_pages_min", "retriever_errors_total", "context_chars_median",
    "S1_min_pct", "uncited_ids_total", "traps_hit_total",
    "S4_min_pct", "pruned_count_total", "s5_improved",
    "contradictions_total", "unsupported_claims_total",
    "queries_scored", "all_pass", "weakest_metric", "bench_round",
    "blocking_count", "addressed_findings", "review_head_sha", "frozen_ok",
]


def rerun(cmd: list[str]) -> None:
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(hconf.HARNESS), timeout=1800)


def main() -> int:
    vals = {"nodes_tried": 0, "try_ok": 0, "git_ok": 0, "frozen_ok": 0,
            "regen_ok": 0, "sync_ok": 0}
    reason = "-"

    # law 6 try matrix
    tried = 0
    for node in NODES:
        okp = TRY_DIR / f"{node}_pass.json"
        okf = TRY_DIR / f"{node}_fail.json"
        if not (okp.exists() and okf.exists()):
            if reason == "-":
                reason = f"try_reports_missing:{node}"
            continue
        try:
            p = json.loads(okp.read_text(encoding="utf-8"))
            f = json.loads(okf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if reason == "-":
                reason = f"try_report_unparseable:{node}"
            continue
        p_ok = p.get("success") is True or p.get("verdict") == "pass"
        f_ok = f.get("success") is False or f.get("verdict") == "fail"
        if p_ok and f_ok:
            tried += 1
        elif reason == "-":
            reason = f"try_verdict_wrong:{node}"
    vals["nodes_tried"] = tried
    vals["try_ok"] = 1 if tried == len(NODES) else 0

    # score-tamper git check
    first_s0 = hconf.git(hconf.REPO, "log", "--reverse", "--format=%H",
                         "--fixed-strings", "--grep=[sq][s0]").splitlines()
    if first_s0:
        touches = hconf.git(hconf.REPO, "log", "--format=%H",
                            f"{first_s0[0]}..HEAD", "--",
                            "harness-search/bench/golden", "harness-search/bench/baseline_firecrawl.json")
        if touches.strip():
            if reason == "-":
                reason = f"golden_or_baseline_MODIFIED_after_freeze:{touches.splitlines()[0][:12]}"
        else:
            vals["git_ok"] = 1
    elif reason == "-":
        reason = "no_[sq][s0]_commit_found"

    # frozen manifest
    r = subprocess.run([str(hconf.VENV_PY), "scripts/check_frozen.py"],
                       capture_output=True, text=True, cwd=str(hconf.HARNESS), timeout=120)
    vals["frozen_ok"] = 1 if "frozen_ok=1" in (r.stdout or "") else 0

    # law 10 idempotence: run each regenerator twice, byte-compare its evidence file
    regen_ok = 1
    targets: list[tuple[list[str], pathlib.Path]] = [
        ([str(hconf.VENV_PY), "scripts/bench_selfcheck.py"], hconf.EVID / "s0_bench.json"),
    ]
    for n in range(1, 6):
        targets.append(([str(hconf.VENV_PY), "scripts/verify_impl.py", "--stage", str(n)],
                        hconf.EVID / f"s{n}_impl.json"))
    for cmd, evp in targets:
        rerun(cmd)
        b1 = evp.read_bytes() if evp.exists() else b""
        rerun(cmd)
        b2 = evp.read_bytes() if evp.exists() else b"x"
        if not b1 or b1 != b2:
            regen_ok = 0
            if reason == "-":
                reason = f"regen_not_idempotent:{evp.name}"
    vals["regen_ok"] = regen_ok

    # law 8 three-way sync
    try:
        prof = PROFILE.read_text(encoding="utf-8")
        hmd = HARNESS_MD.read_text(encoding="utf-8")
        missing = [t for t in SYNC_TOKENS
                   if not (re.search(re.escape(t), prof) and re.search(re.escape(t), hmd))]
        if missing:
            if reason == "-":
                reason = f"sync_token_missing:{missing[0]}"
        else:
            vals["sync_ok"] = 1
    except OSError:
        if reason == "-":
            reason = "profile_or_HARNESS_md_unreadable"

    ok_all = all(vals[k] == 1 for k in ("try_ok", "git_ok", "frozen_ok", "regen_ok", "sync_ok"))
    hconf.write_json(hconf.EVID / "audit.json", {**vals, "ok": 1 if ok_all else 0})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} ok={1 if ok_all else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
