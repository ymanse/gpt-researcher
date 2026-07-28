"""Capture the FROZEN offline corpus: one live deep_tree_research per golden, keeping the
node state and the scraped documents the roll-up needs.

    python scripts/capture.py [--only <golden_id>] [--recapture]

Writes no_read/dedup/corpus/:
    <gid>.tree.json      the artifact the frozen scorer reads (unchanged contract)
    <gid>.report.md      the report the LIVE run produced — the fidelity reference
    <gid>.resynth.json   {"read_docs":{url:text}, "nodes":[...]} — everything the
                         roll-up + report assembly needs, in self.nodes insertion order
    corpus.json          manifest: code_fp at capture + per-query sha256/counts

Why this exists: the roll-up is LLM-only (no retriever, no scrape, no embedding call —
create_chat_completion appears exactly twice in tree_research.py, both inside
research_node/generate_child_questions), so once the node answers and the scraped
documents are on disk the whole synthesis can be re-run for free. A dedup iteration then
costs minutes instead of ~330 Firecrawl credits and 12 minutes PER QUERY.

The corpus is FROZEN on purpose, exactly like bench/golden: the roll-up is what is under
test, so its input must not move underneath the measurement. code_fp is recorded, and a
partial capture that spans an implementation change is refused rather than silently mixed.
Re-capturing is a deliberate, credit-spending act (--recapture) — it is what the d2-live
gate demands when the offline numbers turn out not to hold live.

FOREGROUND ONLY. This takes hours; a `claude -p` session that backgrounds it ends and
orphans the run (measured: 8 iterations lost that way on the search-quality build).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import code_fp
import hconf
import live_lib

CORPUS = hconf.HARNESS / "no_read" / "dedup" / "corpus"
MANIFEST = CORPUS / "corpus.json"


def resynth_host_path(tree_host: str) -> str:
    """<stem>.tree.json -> <stem>.resynth.json (the _persist sidecar, same stem)."""
    p = pathlib.Path(tree_host)
    if p.name.endswith(".tree.json"):
        return str(p.with_name(p.name[: -len(".tree.json")] + ".resynth.json"))
    return ""


def have(gid: str) -> bool:
    return all((CORPUS / f"{gid}{suf}").exists()
               for suf in (".tree.json", ".report.md", ".resynth.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="capture a single golden id (fail-fast probe)")
    ap.add_argument("--recapture", action="store_true",
                    help="discard the existing corpus and re-run every query live")
    a = ap.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)
    fp = code_fp.fingerprint()

    old = {}
    if MANIFEST.exists() and not a.recapture:
        try:
            old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
    if a.recapture:
        for p in CORPUS.glob("*"):
            p.unlink()
        old = {}
    if old and old.get("code_fp") not in (None, fp):
        print(f"capture_ok=0 reason=corpus_code_fp_{old.get('code_fp')}_!=_current_{fp} "
              f"— the implementation moved mid-capture; re-run with --recapture "
              f"(the corpus must be one consistent code version)")
        return 0

    goldens = [g for g in hconf.load_golden()
               if not a.only or g["id"] == a.only]
    if not goldens:
        print(f"capture_ok=0 reason=no_golden_matching:{a.only}")
        return 0

    live_lib.acquire_live_lock()
    todo = [g for g in goldens if not have(g["id"])]
    ok = health = 0
    if todo:
        ok = 1 if live_lib.recreate() else 0
        health = live_lib.wait_health() if ok else 0
        if health != 200:
            print(f"capture_ok=0 recreated={ok} health={health} "
                  f"reason=container_not_healthy — check docker logs {hconf.CONTAINER}")
            return 0
    else:
        ok, health = 1, 200   # nothing to run; the existing corpus is what is verified

    errors: list[str] = []
    for g in goldens:
        gid = g["id"]
        if have(gid):
            print(f"[capture] {gid}: already captured, skipping", flush=True)
            continue
        print(f"[capture] {gid}: running live (this costs credits and ~12 min)", flush=True)
        try:
            res = live_lib.mcp_call("deep_tree_research", {"query": g["query"]}, 5400)
        except BaseException as e:  # noqa: BLE001 — evidence must record the real leaf error
            errors.append(f"{gid}: {live_lib.exc_summary(e)}")
            continue
        tree_host = res.get("tree_json_path") or res.get("tree_path") or ""
        rep_host = res.get("report_path") or res.get("report_md_path") or ""
        rs_host = resynth_host_path(tree_host)
        if not tree_host or not rep_host:
            errors.append(f"{gid}: mcp result lacked tree/report paths: {str(res)[:200]}")
            continue
        if not rs_host or not pathlib.Path(rs_host).exists():
            errors.append(f"{gid}: no {rs_host or '<stem>.resynth.json'} beside the tree — "
                          f"_persist does not emit the resynth sidecar yet")
            continue
        try:
            shutil.copyfile(tree_host, CORPUS / f"{gid}.tree.json")
            shutil.copyfile(rep_host, CORPUS / f"{gid}.report.md")
            shutil.copyfile(rs_host, CORPUS / f"{gid}.resynth.json")
        except OSError as e:
            errors.append(f"{gid}: copy failed: {e}")

    rows = []
    for g in goldens if not a.only else hconf.load_golden():
        gid = g["id"]
        if not have(gid):
            continue
        try:
            rs = json.loads((CORPUS / f"{gid}.resynth.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"{gid}: resynth sidecar unparseable")
            continue
        nodes = rs.get("nodes") or []
        rows.append({
            "id": gid,
            "report_sha256": hconf.sha256(CORPUS / f"{gid}.report.md"),
            "report_chars": len((CORPUS / f"{gid}.report.md").read_text(
                encoding="utf-8", errors="replace")),
            "resynth_sha256": hconf.sha256(CORPUS / f"{gid}.resynth.json"),
            "read_docs_urls": len(rs.get("read_docs") or {}),
            "node_count": len(nodes),
            "kept_nodes": sum(1 for n in nodes
                              if str(n.get("status")) in ("expanded", "answered")),
        })

    man = {
        "code_fp": old.get("code_fp") or fp,
        "golden_count": len(hconf.load_golden()),
        "captured_queries": len(rows),
        "recreated": bool(ok),
        "health": health,
        "queries": sorted(rows, key=lambda r: r["id"]),
        "errors": errors,
    }
    hconf.write_json(MANIFEST, man)
    print(json.dumps({k: v for k, v in man.items() if k != "queries"},
                     indent=2, sort_keys=True))
    for r in man["queries"]:
        print(f"  {r['id'][:24]:26} nodes {r['node_count']:3} kept {r['kept_nodes']:3} "
              f"read_docs {r['read_docs_urls']:3} report {r['report_chars']:6}c")
    print(f"capture_ok={1 if not errors and rows else 0} captured={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
