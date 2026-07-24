"""Stage 2 live smoke: deep_research must yield >=6 learnings per extraction (baseline 3).
Evidence field the gate reads: learnings_count = MAX over TIERA stage=2 lines."""
from __future__ import annotations

import smoke_lib


def main() -> int:
    ev = smoke_lib.smoke_preamble(2)
    if ev["health"] == 200:
        ts = smoke_lib.utc_now_iso()
        try:
            smoke_lib.mcp_call("deep_research", {"query": ev["query"]}, timeout_s=3000)
            ev["mcp_ok"] = True
        except Exception as e:
            ev["mcp_ok"] = False
            ev["mcp_error"] = str(e)[:400]
        lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts), 2)
        if lines:
            ev["learnings_count"] = max(l.get("learnings_count", 0) for l in lines)
            ev["extraction_calls"] = len(lines)
    smoke_lib.finish(2, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
