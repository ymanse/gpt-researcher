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

import argparse
import json
import os
import shutil
import subprocess

import code_fp
import credits
import hconf
import live_lib
import rollup_scan

DEDUP = hconf.HARNESS / "no_read" / "dedup"
CORPUS = DEDUP / "corpus"
OFFLINE = DEDUP / "offline"
RESYNTH = hconf.HARNESS / "scripts" / "resynth.py"


FRONTIER = "(unexplored frontier)"


def baseline_s2(gid: str) -> int | None:
    """S2 of the concatenating report with its UNEXPLORED-FRONTIER lines removed.

    synthesize_node emits "(unexplored frontier) {question}" for every PENDING node, and
    the captured reports carry 16-22 such lines. They are questions nobody researched, not
    findings — but the frozen scorer matches a golden fact pattern anywhere in the file, so
    a report can score a fact purely by reciting the question that mentions it. Measured
    2026-07-29: edge-ai-face-access baseline 75 -> 62 once those lines are excluded, and
    the merge's entire "-13 fact loss" on that query was this artifact. Comparing a merge
    that drops frontier lines (correctly) against a baseline credited for them punishes the
    right behaviour, so the baseline is scored on the stripped text.

    Still the FROZEN scorer, on a real file, via its own CLI — the metric is never
    re-implemented here.
    """
    src = CORPUS / f"{gid}.report.md"
    if not src.exists():
        return None
    stripped = CORPUS / f"{gid}.baseline.report.md"
    text = src.read_text(encoding="utf-8", errors="replace")
    stripped.write_text(
        "\n".join(ln for ln in text.split("\n") if FRONTIER not in ln), encoding="utf-8")
    sc = live_lib.score(gid, str(stripped), str(CORPUS / f"{gid}.tree.json"),
                        f"no_read/dedup/corpus/{gid}.baseline.scores.json")
    return int(sc.get("S2_pct", -1)) if sc else None


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
    ap = argparse.ArgumentParser()
    # A probe run for the impl lane. The merge asks a model for a verdict per candidate
    # pair and a real tree offers 100-300 of them, so re-synthesising all five goldens is
    # the dominant cost of an s9-impl session — two of them hit the 75m agent timeout
    # before they could submit. --only re-synthesises one golden and writes to a
    # SEPARATE evidence file, so a probe can never be mistaken for the d1 gate's evidence
    # (which requires queries_scanned == 5 and would read a partial file as a hard fail).
    ap.add_argument("--only", help="probe a single golden id; writes d1_offline.probe.json")
    a = ap.parse_args()

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
        if a.only and gid != a.only:
            continue
        # the scorer and the lift scan both read the ORIGINAL tree: the node answers are
        # the fixed input, the report is what the merge changed
        shutil.copyfile(CORPUS / f"{gid}.tree.json", OFFLINE / f"{gid}.tree.json")
        (OFFLINE / f"{gid}.scores.json").unlink(missing_ok=True)
        err = resynth(gid)
        if err:
            errors.append(err)
    after = credits.remaining()

    rows, s2s, deltas = [], [], []
    goldens = [g for g in hconf.load_golden() if not a.only or g["id"] == a.only]
    for g in goldens:
        gid = g["id"]
        row = rollup_scan.scan_query(gid, OFFLINE)
        if not row:
            errors.append(f"{gid}: no re-synthesised report/tree in {OFFLINE.name}/")
            continue
        s2 = rollup_scan.s2_from_scores(gid, OFFLINE) or rollup_scan.rescore(gid, OFFLINE)
        # Per-query baseline: the frozen scorer on the CONCATENATING report this merge
        # replaces, scored once off the frozen corpus and cached beside it. The aggregate
        # alone hides the failure that matters — measured 2026-07-28, a merge scored 83
        # aggregate while losing 13 and 12 points on the two queries whose baseline was
        # already the weakest (63 and 75). A mean over five queries lets one report lose
        # half its facts, which is exactly the deletion this gate exists to refuse.
        base = baseline_s2(gid)
        if s2 is None:
            errors.append(f"{gid}: frozen scorer produced no S2")
        elif base is None:
            errors.append(f"{gid}: frozen scorer produced no baseline S2 for the captured report")
        else:
            s2s.append(s2)
            row["S2_pct"] = s2
            row["S2_base_pct"] = base
            row["S2_delta"] = s2 - base
            deltas.append(s2 - base)
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
        # the decisive fact-retention field: worst per-query loss against that query's own
        # concatenating baseline. -999 when a baseline is missing (law 4: unknown != zero)
        "s2_min_delta": min(deltas) if deltas and len(deltas) == len(goldens) else -999,
        "per_query": rows,
        "errors": errors,
    }
    out = hconf.EVID / ("d1_offline.probe.json" if a.only else "d1_offline.json")
    hconf.write_json(out, ev)
    if a.only:
        print(f"PROBE ONLY ({a.only}) -> {out.name}. The d1 gate reads d1_offline.json and "
              f"requires queries_scanned == 5; run without --only before submitting.")
    print(json.dumps({k: v for k, v in ev.items() if k != "per_query"}, indent=2, sort_keys=True))
    for r in rows:
        print(f"  {r['qid'][:22]:24} lifted {r['lifted_nodes']}/{r['node_answers_scanned']} "
              f"max_lift {r['max_lift_pct']}% synth_ratio {r['synthesis_ratio_pct']}% "
              f"S2 {r.get('S2_pct', '-')}(base {r.get('S2_base_pct', '-')}, "
              f"d{r.get('S2_delta', '-'):+}) headings {r['headings']}"
              if "S2_delta" in r else
              f"  {r['qid'][:22]:24} lifted {r['lifted_nodes']}/{r['node_answers_scanned']} "
              f"max_lift {r['max_lift_pct']}% synth_ratio {r['synthesis_ratio_pct']}% "
              f"S2 - headings {r['headings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
