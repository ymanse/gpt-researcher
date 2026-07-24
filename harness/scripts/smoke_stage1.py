"""Stage 1 live smoke: quick_search must route through firecrawl with clean-markdown bodies.
Evidence fields the gate reads: firecrawl_sources>=1, max_firecrawl_body_len>400."""
from __future__ import annotations

import smoke_lib


def main() -> int:
    ev = smoke_lib.smoke_preamble(1)
    if ev["health"] == 200:
        ts = smoke_lib.utc_now_iso()
        try:
            res = smoke_lib.mcp_call("quick_search", {"query": ev["query"]}, timeout_s=900)
            ev["mcp_ok"] = True
            ev["response_keys"] = sorted(res.keys()) if isinstance(res, dict) else []
        except Exception as e:  # evidence stays fail-closed: fields simply absent
            ev["mcp_ok"] = False
            ev["mcp_error"] = str(e)[:400]
        lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts), 1)
        if lines:
            ev["firecrawl_sources"] = max(l.get("firecrawl_results", 0) for l in lines)
            ev["max_firecrawl_body_len"] = max(l.get("max_body_len", 0) for l in lines)
    smoke_lib.finish(1, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
