"""Stage 4 live smoke: academic lane on Firecrawl research (papers + citers expansion).
Evidence fields the gate reads: papers_count>=3, papers_in_window>=1, from_citers_count>=1."""
from __future__ import annotations

import smoke_lib


def main() -> int:
    ev = smoke_lib.smoke_preamble(4)
    if ev["health"] == 200:
        ts = smoke_lib.utc_now_iso()
        try:
            smoke_lib.mcp_call("quick_search", {"query": ev["query"]}, timeout_s=900)
            ev["mcp_ok"] = True
        except Exception as e:
            ev["mcp_ok"] = False
            ev["mcp_error"] = str(e)[:400]
        lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts), 4)
        if lines:
            last = lines[-1]
            for k in ("papers_count", "papers_in_window", "from_citers_count"):
                ev[k] = int(last.get(k, 0))
    smoke_lib.finish(4, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
