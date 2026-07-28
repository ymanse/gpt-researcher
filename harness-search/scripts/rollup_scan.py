"""Roll-up redundancy scan — the ONLY writer of no_read/evidence/s7_dedup.json.

    python scripts/rollup_scan.py [--round N] [--dir <dir with <id>.report.md/.tree.json>]

Measures whether the final report SYNTHESIZES the tree or just concatenates it, and
binds that to fact retention so "dedup" can never be achieved by deleting content.

Why not a sentence/paragraph near-duplicate scan: measured on the round-4 reports, token
Jaccard dedup finds ~0% (repeated 5-grams 0-2%, max section-pair overlap 0.22) while a
reader plainly sees the same themes restated. The redundancy is not lexical — each pasted
node answer is internally unique prose, and sibling nodes overlap in TOPIC, not wording.
What is lexically measurable, and is the actual defect, is the paste itself:

  lifted_nodes        node answers whose 5-grams are >=LIFT_PCT present in the report,
                      i.e. carried over near-verbatim instead of being merged. Measured
                      before any fix: 5-10 of ~13 scanned node answers PER QUERY were
                      100% lifted.
  synthesis_ratio_pct len(report) / sum(len(node answers)) * 100. A real synthesis merges
                      overlapping findings, so this falls well below the measured
                      47-89% baseline. outbox at 89% with 10 fully-lifted nodes is
                      concatenation with a preamble.
  s2_aggregate_pct    facts recall from the FROZEN scorer, carried into this evidence so
                      the gate can refuse a report that got shorter by dropping facts.
                      Deleting content is the obvious way to game a redundancy metric.

Positive bindings (law 1/4): queries_scanned and node_answers_scanned_total must be
non-zero — an empty scan must never read as "no redundancy".

Deterministic and offline: reads cached artifacts only. No LLM, no network, no Firecrawl
credits, so an iteration costs seconds instead of a benchmark round.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import hconf

LIFT_PCT = 70          # a node answer this present in the report was carried, not merged
MIN_NGRAMS = 50        # ignore stub answers; a short node cannot be "lifted" meaningfully
_W = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[[^\]]*\]")


def toks(text: str) -> list[str]:
    return [w for w in _W.findall(_CITE.sub(" ", text.lower())) if len(w) > 3]


def ngrams(ws: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def scan_query(gid: str, d: pathlib.Path) -> dict | None:
    rep_p, tree_p = d / f"{gid}.report.md", d / f"{gid}.tree.json"
    if not rep_p.exists() or not tree_p.exists():
        return None
    report = rep_p.read_text(encoding="utf-8", errors="replace")
    rep_ng = ngrams(toks(report))
    try:
        nodes = json.loads(tree_p.read_text(encoding="utf-8")).get("nodes", [])
    except json.JSONDecodeError:
        return None
    lifted, scanned, total_chars, kept_chars = 0, 0, 0, 0
    worst = 0
    for n in nodes:
        answer = str(n.get("answer") or "")
        total_chars += len(answer)
        # PRUNED/PENDING answers are excluded from the roll-up BY DESIGN, so they must
        # not sit in the denominator: measured against all nodes the ratio rewards a
        # tree for pruning more (edge-ai scored "best" at 47% purely because it pruned
        # 8 nodes). Against the KEPT nodes the real picture appears — the report runs
        # 120-133% of them, i.e. every kept answer plus a preamble, no merging at all.
        if str(n.get("status")) in ("expanded", "answered"):
            kept_chars += len(answer)
        g = ngrams(toks(answer))
        if len(g) < MIN_NGRAMS:
            continue
        scanned += 1
        pct = round(100 * len(g & rep_ng) / len(g))
        worst = max(worst, pct)
        if pct >= LIFT_PCT:
            lifted += 1
    return {
        "qid": gid,
        "node_answers_scanned": scanned,
        "lifted_nodes": lifted,
        "max_lift_pct": worst,
        "report_chars": len(report),
        "node_answer_chars": total_chars,
        "kept_answer_chars": kept_chars,
        # the gated number: report vs the answers the roll-up is actually allowed to use
        "synthesis_ratio_pct": round(100 * len(report) / kept_chars) if kept_chars else 999,
        # informational only — kept for continuity with the first measurement
        "ratio_vs_all_nodes_pct": round(100 * len(report) / total_chars) if total_chars else 999,
        "headings": len(re.findall(r"^#{1,6}\s+\S", report, re.M)),
    }


def s2_from_scores(gid: str, d: pathlib.Path) -> int | None:
    p = d / f"{gid}.scores.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("S2_pct", -1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def rescore(gid: str, d: pathlib.Path) -> int | None:
    """Score S2 with the FROZEN scorer when the directory has no scores yet (a
    re-synthesised report). Never re-implements the metric here."""
    out = d / f"{gid}.scores.json"
    r = subprocess.run(
        [str(hconf.VENV_PY), str(hconf.BENCH / "score_report.py"),
         "--golden", str(hconf.GOLDEN / f"{gid}.json"),
         "--report", str(d / f"{gid}.report.md"),
         "--tree", str(d / f"{gid}.tree.json"),
         "--out", str(out), "--fetch-cache", str(hconf.FETCH_CACHE)],
        capture_output=True, text=True, cwd=str(hconf.HARNESS), timeout=1800,
    )
    return s2_from_scores(gid, d) if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--dir")
    a = ap.parse_args()
    d = pathlib.Path(a.dir) if a.dir else hconf.BENCH_RUNS / f"round{a.round if a.round is not None else hconf.bench_round()}"

    rows, s2s, errors = [], [], []
    for g in hconf.load_golden():
        gid = g["id"]
        row = scan_query(gid, d)
        if not row:
            errors.append(f"{gid}: report/tree missing in {d.name}")
            continue
        s2 = s2_from_scores(gid, d)
        if s2 is None:
            s2 = rescore(gid, d)
        if s2 is None:
            errors.append(f"{gid}: no S2 score")
        else:
            s2s.append(s2)
            row["S2_pct"] = s2
        rows.append(row)

    ev = {
        "stage": 7, "phase": "dedup", "source_dir": d.name,
        "code_fp": __import__("code_fp").fingerprint(),
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
    hconf.write_json(hconf.EVID / "s7_dedup.json", ev)
    print(json.dumps({k: v for k, v in ev.items() if k != "per_query"}, indent=2, sort_keys=True))
    for r in rows:
        print(f"  {r['qid'][:22]:24} lifted {r['lifted_nodes']}/{r['node_answers_scanned']} "
              f"max_lift {r['max_lift_pct']}% synth_ratio {r['synthesis_ratio_pct']}% "
              f"S2 {r.get('S2_pct','-')} headings {r['headings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
