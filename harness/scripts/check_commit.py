"""COMMIT gate check — run IN-GATE by smoke_common.lua and final_verify.py.

    python scripts/check_commit.py --stage N

For gpt-researcher (and gptr-mcp when stage in 5,6): the repo must be on
feature/tier-a-upgrade, a commit whose message contains the literal [tier-a][stageN]
must exist on HEAD's history, and the TRACKED tree must be clean (all changes committed;
untracked files are ignored — no_read/, .gralph/, .omc/ are gitignored anyway).
Prints: commits_ok=1 reason=-   or   commits_ok=0 reason=<prescriptive>
"""
from __future__ import annotations

import argparse

import hconf


def check(stage: int) -> tuple[bool, str]:
    repos = [("gpt-researcher", hconf.REPO)]
    if stage in (5, 6):
        repos.append(("gptr-mcp", hconf.MCP_REPO))
    tag = f"[tier-a][stage{stage}]"
    for label, repo in repos:
        branch = hconf.git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if branch != hconf.BRANCH:
            return False, f"{label} is on '{branch}', not {hconf.BRANCH}"
        log = hconf.git(repo, "log", "--oneline", "--fixed-strings", f"--grep={tag}")
        if not log:
            return False, f"no commit containing '{tag}' in {label}"
        porcelain = hconf.git(repo, "status", "--porcelain")
        dirty = [ln for ln in porcelain.splitlines() if ln and not ln.startswith("??")]
        if dirty:
            return False, f"{label} has {len(dirty)} uncommitted tracked change(s)"
    return True, "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 7))
    a = ap.parse_args()
    ok, reason = check(a.stage)
    print(f"commits_ok={1 if ok else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
