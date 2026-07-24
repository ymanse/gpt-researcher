"""Stage 5 live smoke: scope=True builds a brief; scope=False stays regression-free.
Evidence fields the gate reads: scope_on_brief_present=true, brief_len>=1, scope_off_ok=true."""
from __future__ import annotations

import smoke_lib


def main() -> int:
    ev = smoke_lib.smoke_preamble(5)
    if ev["health"] == 200:
        # ON call — must emit TIERA_EVIDENCE stage=5 brief line
        ts_on = smoke_lib.utc_now_iso()
        try:
            smoke_lib.mcp_call("deep_research", {"query": ev["query"], "scope": True}, timeout_s=3000)
            ev["scope_on_mcp_ok"] = True
        except Exception as e:
            ev["scope_on_mcp_ok"] = False
            ev["mcp_error_on"] = smoke_lib.exc_summary(e)
        on_lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts_on), 5)
        if on_lines:
            last = on_lines[-1]
            ev["scope_on_brief_present"] = bool(int(last.get("brief_present", 0)))
            ev["brief_len"] = int(last.get("brief_len", 0))

        # OFF call — must complete like today, with NO stage=5 brief line in its window
        ts_off = smoke_lib.utc_now_iso()
        try:
            smoke_lib.mcp_call("deep_research", {"query": ev["query"], "scope": False}, timeout_s=3000)
            off_ok = True
        except Exception as e:
            off_ok = False
            ev["mcp_error_off"] = smoke_lib.exc_summary(e)
        off_lines = smoke_lib.tiera_lines(smoke_lib.docker_logs_since(ts_off), 5)
        ev["scope_off_ok"] = bool(off_ok and not off_lines)
    smoke_lib.finish(5, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
