"""Stage 3 live smoke: citation-verification pass over a live deep_research run.
Evidence fields the gate reads: total_claims>=1, grounded_ratio_pct>=70, unverified present."""
from __future__ import annotations

import smoke_lib


def main() -> int:
    ev = smoke_lib.smoke_preamble(3)
    if ev["health"] == 200:
        ts = smoke_lib.utc_now_iso()
        try:
            smoke_lib.mcp_call("deep_research", {"query": ev["query"]}, timeout_s=3000)
            ev["mcp_ok"] = True
        except Exception as e:
            ev["mcp_ok"] = False
            ev["mcp_error"] = str(e)[:400]
        lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts), 3)
        if lines:
            last = lines[-1]
            total = int(last.get("total_claims", 0))
            grounded = int(last.get("grounded", 0))
            ev["total_claims"] = total
            ev["grounded"] = grounded
            ev["unverified"] = int(last.get("unverified", 0))
            ev["grounded_ratio_pct"] = (grounded * 100 // total) if total else 0
    smoke_lib.finish(3, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
