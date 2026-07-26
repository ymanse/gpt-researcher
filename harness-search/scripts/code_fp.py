"""Implementation fingerprint — the cache key and the measure gate's freshness proof.

    python scripts/code_fp.py      -> prints e.g.  code_fp=3f9a1c0b7d2e4a55

sha256 over the CONTENT of every implementation file whose bytes can change what a live
tree run produces: gpt_researcher/**/*.py plus gptr-mcp/server.py. Content-hashed rather
than keyed on a git sha because agents iterate BEFORE committing — a HEAD-based key would
happily reuse a tree produced by code that has since been edited, and re-scoring a stale
artifact after a fix makes the measure gate unsatisfiable (measured: s2 round 0).

Deterministic and idempotent: same bytes -> same fingerprint.
"""
from __future__ import annotations

import hashlib

import hconf


def fingerprint() -> str:
    h = hashlib.sha256()
    files = sorted(hconf.REPO.joinpath("gpt_researcher").rglob("*.py"))
    files.append(hconf.MCP_REPO / "server.py")
    for p in files:
        if not p.exists():
            continue
        h.update(p.as_posix().encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(f"code_fp={fingerprint()}")
