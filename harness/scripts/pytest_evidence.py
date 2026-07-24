"""Emit RED-phase evidence for a stage from pytest's own junit output (law 10).

    python scripts/pytest_evidence.py --stage N --phase red

Runs pytest on tests/tier_a/stageN, writes no_read/evidence/stageN_red.{xml,json}.
The JSON records collected/errors/passed/failed/skipped plus sha256 of every test file —
the GREEN gate later recomputes those hashes from disk to prove tests weren't weakened.
Idempotent: re-running against the same tree yields identical JSON bytes.
"""
from __future__ import annotations

import argparse

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--phase", required=True, choices=["red"])
    a = ap.parse_args()

    stage_dir = hconf.TESTS / f"stage{a.stage}"
    junit = hconf.EVID / f"stage{a.stage}_red.xml"
    counts = hconf.run_pytest(stage_dir, junit)
    test_files = {
        str(p.relative_to(hconf.REPO)).replace("\\", "/"): hconf.sha256(p)
        for p in sorted(stage_dir.glob("test_*.py"))
    } if stage_dir.exists() else {}

    ev = {
        "stage": a.stage,
        "phase": "red",
        **counts,
        "test_files": test_files,
        "junit": f"no_read/evidence/stage{a.stage}_red.xml",
    }
    out = hconf.EVID / f"stage{a.stage}_red.json"
    hconf.write_json(out, ev)
    print(f"wrote {out}: collected={counts['collected']} errors={counts['errors']} "
          f"passed={counts['passed']} failed={counts['failed']} files={len(test_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
