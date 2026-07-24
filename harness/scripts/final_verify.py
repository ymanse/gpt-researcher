"""Capstone verifier — run IN-GATE by final_verify.lua.

    python scripts/final_verify.py

Recomputes: full tests/tier_a suite (junit-derived), presence + stage-tag of all 18
evidence files, and all 6 stage commits (both repos where applicable).
Prints ONE machine line; writes no_read/evidence/final.json.
"""
from __future__ import annotations

import json

import check_commit
import hconf


def main() -> int:
    suite = hconf.run_pytest(hconf.TESTS, hconf.EVID / "final_suite.xml")

    evidence_ok, evidence_reason = 1, "-"
    for n in range(1, 7):
        for kind in ("red", "green", "smoke"):
            p = hconf.EVID / f"stage{n}_{kind}.json"
            if not p.exists():
                evidence_ok, evidence_reason = 0, f"missing:stage{n}_{kind}.json"
                break
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("stage") != n:
                    evidence_ok, evidence_reason = 0, f"mistagged:stage{n}_{kind}.json"
                    break
            except json.JSONDecodeError:
                evidence_ok, evidence_reason = 0, f"unparseable:stage{n}_{kind}.json"
                break
        if not evidence_ok:
            break

    commits_ok, commit_reason = 1, "-"
    for n in range(1, 7):
        ok, reason = check_commit.check(n)
        if not ok:
            commits_ok, commit_reason = 0, f"stage{n}: {reason}"
            break

    vals = {f"suite_{k}": v for k, v in suite.items()}
    vals.update({"evidence_ok": evidence_ok, "commits_ok": commits_ok})
    hconf.write_json(hconf.EVID / "final.json",
                     {**vals, "evidence_reason": evidence_reason, "commit_reason": commit_reason})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} evidence_reason={evidence_reason}| commit_reason={commit_reason}|")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
