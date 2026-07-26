"""COMMIT gate check — run IN-GATE by measure/benchmark gates.

    python scripts/check_commit.py --stage N     (N in 0..6)

Both repos (gpt-researcher, gptr-mcp) must be on feature/search-quality with a CLEAN
tracked tree (all changes committed; untracked ignored — no_read/, .gralph/ are
gitignored). gpt-researcher additionally needs a commit containing [sq][sN] on HEAD's
history. gptr-mcp only needs the tag if the stage touched it (clean-tree covers it).
Prints: commits_ok=1 reason=-   or   commits_ok=0 reason=<prescriptive>
"""
from __future__ import annotations

import argparse

import hconf


def check(stage: int) -> tuple[bool, str]:
    tag = f"{hconf.TAG}[s{stage}]"
    for label, repo, need_tag in (("gpt-researcher", hconf.REPO, True),
                                  ("gptr-mcp", hconf.MCP_REPO, False)):
        branch = hconf.git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if branch != hconf.BRANCH:
            return False, f"{label} is on '{branch}', not {hconf.BRANCH}"
        if need_tag:
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
    ap.add_argument("--stage", type=int, required=True, choices=range(0, 7))
    a = ap.parse_args()
    ok, reason = check(a.stage)
    print(f"commits_ok={1 if ok else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
