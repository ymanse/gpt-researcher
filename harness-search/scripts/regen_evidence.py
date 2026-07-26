"""Law-10 regenerator dispatcher — rebuild ANY evidence file from tool truth, idempotently.

    python scripts/regen_evidence.py --what s0|impl|measure|benchmark [--stage N]

s0        -> bench_selfcheck.py (re-scores fixtures via fetch cache)
impl      -> verify_impl.py --stage N (re-hashes tests, re-runs suites, re-runs ruff)
measure   -> measure.py --stage N (re-scores from the round cache; only re-runs live
             queries whose cache is missing)
benchmark -> benchmark.py (same round-cache rule)

Running any target twice yields identical bytes. Hand-edited evidence does not survive
this script — that is the point.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["s0", "impl", "measure", "benchmark"])
    ap.add_argument("--stage", type=int, choices=range(1, 6))
    a = ap.parse_args()
    if a.what in ("impl", "measure") and a.stage is None:
        print("--stage required for impl/measure")
        return 2
    cmd = {
        "s0": ["scripts/bench_selfcheck.py"],
        "impl": ["scripts/verify_impl.py", "--stage", str(a.stage)],
        "measure": ["scripts/measure.py", "--stage", str(a.stage)],
        "benchmark": ["scripts/benchmark.py"],
    }[a.what]
    r = subprocess.run([str(hconf.VENV_PY), *cmd], cwd=str(hconf.HARNESS), timeout=7200,
                       stdout=sys.stdout, stderr=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
