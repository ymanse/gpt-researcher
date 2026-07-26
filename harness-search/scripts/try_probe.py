"""Law-6 try recorder — runs `gralph try` and records ITS verdict as tool evidence.

    python scripts/try_probe.py --node s1-red --expect pass \
        --report no_read/audit/try/s1-red_pass.json [-- <node args...>]

gralph try has no report-output flag, so this script is the deterministic recorder:
it invokes `gralph try <node> --profile search-quality.yaml <node args>` as a
subprocess and writes {"node","expect","exit_code","success","stdout_tail"} where
success = (exit_code == 0). audit_check.py reads these files; the probe evidence the
gate judged is whatever was on disk when this ran (the audit agent's known-good /
known-bad files).
"""
from __future__ import annotations

import argparse
import subprocess

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--expect", required=True, choices=["pass", "fail"])
    ap.add_argument("--report", required=True)
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()

    r = subprocess.run(
        ["gralph", "try", a.node, "--profile", "search-quality.yaml", *a.rest],
        capture_output=True, text=True, errors="replace", cwd=str(hconf.HARNESS),
        timeout=3600,
    )
    out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    ev = {
        "node": a.node,
        "expect": a.expect,
        "exit_code": r.returncode,
        "success": r.returncode == 0,
        "stdout_tail": out[-800:],
    }
    hconf.write_json(hconf.HARNESS / a.report, ev)
    print(f"node={a.node} expect={a.expect} exit={r.returncode} "
          f"matched={(r.returncode == 0) == (a.expect == 'pass')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
