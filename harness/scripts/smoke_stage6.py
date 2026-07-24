"""Stage 6 live smoke: deep_tree_research end-to-end; artifacts read from the host outputs
bind mount (D:/dev_ext/gptr-mcp/outputs). Gate fields: tree_json_present, report_md_present,
max_depth_reached>=2, node_count>=3, pruned_count present, citations_resolve, budget_respected."""
from __future__ import annotations

import json
import re
import time

import hconf
import smoke_lib


def _newest(pattern: str, since_epoch: float):
    cands = [p for p in hconf.OUTPUTS_HOST.rglob(pattern) if p.stat().st_mtime >= since_epoch]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main() -> int:
    ev = smoke_lib.smoke_preamble(6)
    if ev["health"] == 200:
        t0 = time.time() - 60  # tolerate clock skew between host and container writes
        try:
            res = smoke_lib.mcp_call("deep_tree_research", {
                "query": ev["query"], "max_depth": 2, "max_nodes": 8, "credit_budget": 30,
            }, timeout_s=3600)
            ev["mcp_ok"] = True
            ev["response_keys"] = sorted(res.keys()) if isinstance(res, dict) else []
        except Exception as e:
            ev["mcp_ok"] = False
            ev["mcp_error"] = str(e)[:400]

        tree_path = _newest("*tree*.json", t0)
        report_path = _newest("*.md", t0)
        ev["tree_json_present"] = tree_path is not None
        report_text = report_path.read_text(encoding="utf-8", errors="replace") if report_path else ""
        ev["report_md_present"] = bool(report_text.strip())

        if tree_path:
            try:
                tree = json.loads(tree_path.read_text(encoding="utf-8"))
                meta = tree.get("meta", {})
                nodes = tree.get("nodes", [])
                citations = tree.get("citations", {})
                ev["node_count"] = int(meta.get("node_count", len(nodes)))
                ev["max_depth_reached"] = int(meta.get(
                    "max_depth_reached", max((n.get("depth", 0) for n in nodes), default=0)))
                ev["pruned_count"] = int(meta.get(
                    "pruned_count", sum(1 for n in nodes if str(n.get("status", "")).upper() == "PRUNED")))
                ev["budget_respected"] = bool(meta.get("budget_respected", False))
                urls_ok = bool(citations) and all(
                    str(u).startswith("http") for u in citations.values())
                cited_ids = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", report_text))
                referenced = cited_ids & set(citations.keys())
                ev["citations_resolve"] = bool(urls_ok and referenced)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                ev["tree_parse_error"] = str(e)[:200]
    smoke_lib.finish(6, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
