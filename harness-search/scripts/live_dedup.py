"""d2 live confirm — the ONLY writer of no_read/evidence/d2_live.json.

    python scripts/live_dedup.py [--golden outbox-failure-modes]

Runs ONE golden end to end through the real container and measures the same roll-up
numbers d1 measured offline, plus the frozen scorer's S1/S2/S3. Two questions, one run:

  1. does the merge actually hold in the live pipeline (not just over cached answers)?
  2. does the OFFLINE measurement tell the truth? ratio_gap / lifted_gap compare this run
     against d1_offline.json for the same query. A gap means d0's fidelity proof has
     stopped holding — the loop has been optimising numbers that live never produces —
     and the gate routes back to d0 rather than accepting the result.

outbox-failure-modes is the default because it was the worst case measured: 12 of 12 kept
node answers carried into the report >=70% verbatim, 129% synthesis ratio, 2 headings,
81k chars.

Cached on (golden, code_fp) so re-running the gate re-reads the artifact instead of
buying a second live run; the moment the implementation changes the run is repeated.

FOREGROUND ONLY — ~12 minutes and ~330 credits. A backgrounded run is orphaned when the
session ends (measured: 8 iterations lost that way).
"""
from __future__ import annotations

import argparse
import json
import shutil

import code_fp
import hconf
import live_lib
import rollup_scan

DEDUP = hconf.HARNESS / "no_read" / "dedup"
LIVE = DEDUP / "live"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="outbox-failure-modes")
    a = ap.parse_args()
    gid = a.golden

    LIVE.mkdir(parents=True, exist_ok=True)
    fp = code_fp.fingerprint()
    tree_p, rep_p, fp_p = LIVE / f"{gid}.tree.json", LIVE / f"{gid}.report.md", LIVE / f"{gid}.codefp"
    cached = (tree_p.exists() and rep_p.exists() and fp_p.exists()
              and fp_p.read_text(encoding="utf-8").strip() == fp)

    ev: dict = {"stage": "d2", "phase": "live", "golden": gid, "code_fp": fp,
                "cached": bool(cached), "recreated": False, "health": 0, "errors": []}

    golden = next((g for g in hconf.load_golden() if g["id"] == gid), None)
    if not golden:
        ev["errors"].append(f"no golden with id {gid}")
        hconf.write_json(hconf.EVID / "d2_live.json", ev)
        print(json.dumps(ev, indent=2, sort_keys=True))
        return 0

    if cached:
        # the artifact on disk was produced by these exact bytes; recreate+health were
        # proven when it was produced and are re-asserted from the cache record
        ev["recreated"], ev["health"] = True, 200
    else:
        live_lib.acquire_live_lock()
        ok = live_lib.recreate()
        ev["recreated"] = bool(ok)
        ev["health"] = live_lib.wait_health() if ok else 0
        if ev["health"] == 200:
            try:
                res = live_lib.mcp_call("deep_tree_research", {"query": golden["query"]}, 5400)
                th = res.get("tree_json_path") or res.get("tree_path") or ""
                rh = res.get("report_path") or res.get("report_md_path") or ""
                if th and rh:
                    shutil.copyfile(th, tree_p)
                    shutil.copyfile(rh, rep_p)
                    fp_p.write_text(fp + "\n", encoding="utf-8")
                else:
                    ev["errors"].append(f"mcp result lacked paths: {str(res)[:200]}")
            except BaseException as e:  # noqa: BLE001 — record the real leaf error
                ev["errors"].append(live_lib.exc_summary(e))
        else:
            ev["errors"].append("container not healthy after force-recreate")

    row = rollup_scan.scan_query(gid, LIVE)
    if row:
        ev["live_lifted_nodes"] = row["lifted_nodes"]
        ev["live_node_answers_scanned"] = row["node_answers_scanned"]
        ev["live_synthesis_ratio_pct"] = row["synthesis_ratio_pct"]
        ev["live_headings"] = row["headings"]
        ev["live_report_chars"] = row["report_chars"]
    else:
        ev["errors"].append(f"no {gid}.report.md/.tree.json in {LIVE.name}/ to scan")

    sc = live_lib.score(gid, str(rep_p), str(tree_p),
                        f"no_read/dedup/live/{gid}.scores.json") if rep_p.exists() else None
    if sc:
        for i in (1, 2, 3):
            ev[f"live_S{i}_pct"] = int(sc.get(f"S{i}_pct", -1))
    else:
        ev["errors"].append("frozen scorer produced no scores")

    # offline<->live agreement for THIS query
    try:
        off = json.loads((hconf.EVID / "d1_offline.json").read_text(encoding="utf-8"))
        orow = next((r for r in off.get("per_query", []) if r.get("qid") == gid), None)
    except (OSError, json.JSONDecodeError):
        orow = None
    if orow and row:
        ev["offline_ratio_pct"] = orow["synthesis_ratio_pct"]
        ev["offline_lifted_nodes"] = orow["lifted_nodes"]
        ev["ratio_gap"] = abs(row["synthesis_ratio_pct"] - orow["synthesis_ratio_pct"])
        ev["lifted_gap"] = abs(row["lifted_nodes"] - orow["lifted_nodes"])
    else:
        # unknown agreement is not agreement (law 4)
        ev["ratio_gap"] = 999
        ev["lifted_gap"] = 999
        ev["errors"].append(f"no d1_offline.json row for {gid} — run offline_dedup.py first")

    hconf.write_json(hconf.EVID / "d2_live.json", ev)
    print(json.dumps(ev, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
