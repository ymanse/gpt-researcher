"""Emit RED-phase evidence for a stage from pytest's own junit output (law 10).

    python scripts/pytest_evidence.py --stage N --phase red

Runs pytest on tests/search_quality/sN, writes no_read/evidence/sN_red.{xml,json}.
The JSON records collected/errors/passed/failed/skipped, sha256 of every test file
(the impl gate later recomputes these from disk to prove tests weren't weakened),
and base_sha = gpt-researcher HEAD at RED time (the review agent diffs against it).
Idempotent: re-running against the same tree yields identical JSON bytes.
"""
from __future__ import annotations

import argparse

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    # 1-5 = search-quality lanes; 9 = the dedup harness's roll-up lane
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 10))
    ap.add_argument("--phase", required=True, choices=["red"])
    # Law-10 recovery ONLY: rebuild sN_red.json from the junit pytest already wrote,
    # instead of re-running pytest. Needed because RED is not reproducible once the
    # implementation is GREEN — a re-run would report passed>0 and the honest evidence
    # would be gone for good. Every field still comes from a tool artifact: the counts
    # from pytest's own report, the hashes from the test files on disk, base_sha from the
    # commit that added them. Fails closed if any of the three is missing.
    ap.add_argument("--from-junit", action="store_true",
                    help="rebuild the JSON from the existing sN_red.xml (do not re-run pytest)")
    a = ap.parse_args()

    stage_dir = hconf.TESTS / f"s{a.stage}"
    junit = hconf.EVID / f"s{a.stage}_red.xml"
    if a.from_junit:
        if not junit.exists():
            print(f"REFUSING: {junit} is gone — the RED counts cannot be recovered from "
                  f"anything else, and re-running pytest against a GREEN tree would record "
                  f"passed>0 as 'red'. Re-cut the RED stage instead.")
            return 1
        counts = hconf.parse_junit(junit)
    else:
        counts = hconf.run_pytest(stage_dir, junit)
    test_files = {
        str(p.relative_to(hconf.REPO)).replace("\\", "/"): hconf.sha256(p)
        for p in sorted(stage_dir.glob("test_*.py"))
    } if stage_dir.exists() else {}

    if a.from_junit:
        # the RED commit is the most recent one that ADDED these test files
        adds = [hconf.git(hconf.REPO, "log", "--diff-filter=A", "--format=%H", "--", rel)
                .splitlines() for rel in sorted(test_files)]
        base = next((ls[0] for ls in adds if ls), "")
        if not base or not test_files:
            print("REFUSING: cannot derive base_sha (no test files, or none of them was "
                  "ever added in a commit) — recover by re-cutting the RED stage")
            return 1
    else:
        base = hconf.git(hconf.REPO, "rev-parse", "HEAD")

    ev = {
        "stage": a.stage,
        "phase": "red",
        **counts,
        "test_files": test_files,
        "base_sha": base,
        "junit": f"no_read/evidence/s{a.stage}_red.xml",
    }
    out = hconf.EVID / f"s{a.stage}_red.json"
    hconf.write_json(out, ev)
    print(f"wrote {out}: collected={counts['collected']} errors={counts['errors']} "
          f"passed={counts['passed']} failed={counts['failed']} files={len(test_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
