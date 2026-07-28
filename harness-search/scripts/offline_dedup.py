"""d1 offline measure — the ONLY writer of no_read/evidence/d1_offline.json.

    python scripts/offline_dedup.py

Re-synthesises every captured golden with the implementation currently on disk, then
measures the roll-up exactly the way the s7 gate does (scripts/rollup_scan.py, reused as a
module so there is ONE definition of "lifted" and one scorer call path).

The point of the whole harness is here: this costs zero Firecrawl credits, so a merge
strategy can be tried, measured and thrown away in minutes instead of the 2.6 hours and
~5,800 credits a live benchmark round costs. credits_delta is read from the vendor's own
balance before and after and must be 0 — "offline" is a claim until something checks it.

What it does NOT prove: that the live pipeline still produces these numbers end to end.
That is d2-live's job, and the two must agree within 10 points or the fidelity proof
d0 established has stopped holding.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import code_fp
import credits
import hconf
import rollup_scan

DEDUP = hconf.HARNESS / "no_read" / "dedup"
CORPUS = DEDUP / "corpus"
OFFLINE = DEDUP / "offline"
RESYNTH = hconf.HARNESS / "scripts" / "resynth.py"


def resynth(gid: str) -> str:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    r = subprocess.run(
        [str(hconf.VENV_PY), str(RESYNTH),
         "--resynth", str(CORPUS / f"{gid}.resynth.json"),
         "--tree", str(CORPUS / f"{gid}.tree.json"),
         "--out", str(OFFLINE), "--as", gid],
        capture_output=True, text=True, errors="replace", env=env,
        cwd=str(hconf.HARNESS), timeout=3600,
    )
    if r.returncode != 0:
        return f"{gid}: resynth.py exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}"
    if not (OFFLINE / f"{gid}.report.md").exists():
        return f"{gid}: resynth.py produced no {gid}.report.md"
    return ""


def main() -> int:
    OFFLINE.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        man = json.loads((CORPUS / "corpus.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        man = {}
        errors.append("no_read/dedup/corpus/corpus.json missing — run scripts/capture.py (d0)")

    before = credits.remaining()
    for r in man.get("queries") or []:
        gid = r["id"]
        # the scorer and the lift scan both read the ORIGINAL tree: the node answers are
        # the fixed input, the report is what the merge changed
        shutil.copyfile(CORPUS / f"{gid}.tree.json", OFFLINE / f"{gid}.tree.json")
        (OFFLINE / f"{gid}.scores.json").unlink(missing_ok=True)
        err = resynth(gid)
        if err:
            errors.append(err)
    after = credits.remaining()

    rows, s2s = [], []
    for g in hconf.load_golden():
        gid = g["id"]
        row = rollup_scan.scan_query(gid, OFFLINE)
        if not row:
            errors.append(f"{gid}: no re-synthesised report/tree in {OFFLINE.name}/")
            continue
        s2 = rollup_scan.s2_from_scores(gid, OFFLINE) or rollup_scan.rescore(gid, OFFLINE)
        if s2 is None:
            errors.append(f"{gid}: frozen scorer produced no S2")
        else:
            s2s.append(s2)
            row["S2_pct"] = s2
        rows.append(row)

    ev = {
        "stage": "d1", "phase": "offline", "source_dir": OFFLINE.name,
        "code_fp": code_fp.fingerprint(),
        "corpus_code_fp": man.get("code_fp", ""),
        # -1 from either read means "balance unknown" -> a non-zero delta -> fail closed
        "credits_before": before, "credits_after": after,
        "credits_delta": (before - after) if (before >= 0 and after >= 0) else 999,
        "resynth_failed": len([e for e in errors if "resynth.py" in e]),
        "queries_scanned": len(rows),
        "node_answers_scanned_total": sum(r["node_answers_scanned"] for r in rows),
        "lifted_nodes_max": max((r["lifted_nodes"] for r in rows), default=999),
        "lifted_nodes_total": sum(r["lifted_nodes"] for r in rows),
        "synthesis_ratio_pct_max": max((r["synthesis_ratio_pct"] for r in rows), default=999),
        "headings_min": min((r["headings"] for r in rows), default=0),
        "s2_aggregate_pct": round(sum(s2s) / len(s2s)) if s2s else 0,
        "per_query": rows,
        "errors": errors,
    }
    hconf.write_json(hconf.EVID / "d1_offline.json", ev)
    print(json.dumps({k: v for k, v in ev.items() if k != "per_query"}, indent=2, sort_keys=True))
    for r in rows:
        print(f"  {r['qid'][:22]:24} lifted {r['lifted_nodes']}/{r['node_answers_scanned']} "
              f"max_lift {r['max_lift_pct']}% synth_ratio {r['synthesis_ratio_pct']}% "
              f"S2 {r.get('S2_pct', '-')} headings {r['headings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
