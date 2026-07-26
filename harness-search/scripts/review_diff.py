"""Print the diff a review agent is allowed to see: everything since the stage's RED
base_sha (committed and uncommitted), gpt-researcher + gptr-mcp.

    python scripts/review_diff.py --stage N

The review agent gets ONLY this diff plus the stage's completion conditions from
spec/search-quality.md — never the implementer's reasoning (separate-lane rule).
"""
from __future__ import annotations

import argparse
import json
import subprocess

import hconf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 6))
    a = ap.parse_args()

    red_path = hconf.EVID / f"s{a.stage}_red.json"
    if not red_path.exists():
        print(f"NO_RED_EVIDENCE: {red_path} missing — the red stage must run first")
        return 1
    base = json.loads(red_path.read_text(encoding="utf-8")).get("base_sha", "")
    if not base:
        print("NO_BASE_SHA in red evidence")
        return 1

    for label, repo, ref in (("gpt-researcher", hconf.REPO, base),
                             ("gptr-mcp", hconf.MCP_REPO, "HEAD")):
        print(f"===== {label} diff ({ref}..working tree) =====")
        r = subprocess.run(["git", "diff", ref], capture_output=True, text=True,
                           cwd=str(repo), errors="replace", timeout=120)
        print(r.stdout or "(no diff)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
