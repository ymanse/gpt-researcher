"""s0 gate verifier — run IN-GATE by s0_bench.lua (and as the s0 agent's preflight).

    python scripts/bench_selfcheck.py

Deterministically verifies the benchmark harness the s0 agent built:
  golden_count       >=5 golden files, schema-complete (query, required_primary_domains,
                     facts, traps, contested, coverage_areas; all regexes compile)
  bun_first          bench/golden contains id "bun-rust-port" whose query matches the
                     observed session artifact (outputs/how-did-bun-port-*)
  measure_pairs      exactly 2 goldens have measure_pair:true, one of them bun-rust-port
  llm_calls          static scan of bench/*.py for LLM SDK imports / chat endpoints /
                     claude-CLI subprocess calls — must be 0 occurrences (law 2: the
                     scorer may never ask a model)
  fixtures_passed    score_report.py run on bench/fixtures/{good,bad}_* — good must beat
                     bad STRICTLY on S1,S2,S4,S5,S6 and be STRICTLY lower on S3 (law 5:
                     proves [] is discrimination, not parse failure). 2 = both runs scored.
  baseline_queries   baseline_firecrawl.json scores every golden id, each entry citing an
                     existing bench/baseline_runs/<id>.md report + scores file (provenance)

Prints ONE machine line; writes no_read/evidence/s0_bench.json. Idempotent (fetch-cache
makes S1 re-scoring stable); re-running is the law-10 regenerator for s0 evidence.
"""
from __future__ import annotations

import json
import re
import subprocess

import hconf

REQUIRED_KEYS = ("id", "query", "category", "required_primary_domains", "facts", "traps",
                 "contested", "coverage_areas")
CATEGORIES = ("recent-event", "technical-deep-dive", "market-landscape",
              "contested-forecast", "academic", "encyclopedic-entity")
# ponytail: substring scan, not AST — these tokens have no legitimate place in a
# deterministic scorer, so a false positive just means "rename your variable".
LLM_TOKENS = ("openai", "anthropic", "litellm", "langchain", "chat.completions",
              "claude -p", "claude\", \"-p", "genai", "cohere", "mistralai",
              "completion(", "ChatCompletion")


def schema_ok(g: dict) -> str:
    for k in REQUIRED_KEYS:
        if k not in g or not g[k]:
            return f"missing_or_empty:{k}"
    try:
        for f in g["facts"]:
            re.compile(f["pattern"])
        for t in g["traps"]:
            re.compile(t["pattern"])
        for c in g["contested"]:
            if len(c["values"]) < 2:
                return "contested_needs_2_values"
            for v in c["values"]:
                re.compile(v)
        for a in g["coverage_areas"]:
            re.compile(a["pattern"])
    except (re.error, KeyError, TypeError) as e:
        return f"bad_regex_or_shape:{e}"
    return ""


def _dated(g: dict) -> bool:
    """Deterministic proxy for prior-knowledge-hostile: at least one fact anchored to a
    2024-2029 date. A benchmark of timeless best-practice queries lets an empty-context
    node answer correctly from prior knowledge, making the trap metric (S3) vacuous."""
    return any(re.search(r"202[4-9]", f.get("pattern", "") + f.get("desc", ""))
               for f in g.get("facts", []))


def diversity(goldens: list[dict]) -> str:
    """Return '' when the set is meaningfully diverse, else a compact reason token."""
    for g in goldens:
        gid = g.get("id", "?")
        if g.get("category") not in CATEGORIES:
            return f"category_invalid:{gid}"
        if len(g["facts"]) < 5:
            return f"facts_lt_5:{gid}"
        if len(g["traps"]) < 3:
            return f"traps_lt_3:{gid}"
        if len(g["coverage_areas"]) < 4:
            return f"areas_lt_4:{gid}"
        if len(g["required_primary_domains"]) < 2:
            return f"domains_lt_2:{gid}"
        if len(g["contested"]) < 1:
            return f"contested_lt_1:{gid}"
    cats = {g["category"] for g in goldens}
    if len(cats) < 4:
        return f"categories_distinct_{len(cats)}_lt_4"
    if sum(1 for g in goldens if _dated(g)) < 2:
        return "dated_goldens_lt_2"
    pair = [g for g in goldens if g.get("measure_pair") is True]
    if len(pair) == 2:
        if pair[0]["category"] == pair[1]["category"]:
            return "measure_pair_same_category"
        if not any(_dated(g) for g in pair):
            return "measure_pair_has_no_dated_golden"
    domains = {d.lower() for g in goldens for d in g["required_primary_domains"]}
    if len(domains) < 6:
        return f"domain_union_{len(domains)}_lt_6"
    return ""


def run_scorer(golden: str, report: str, tree: str | None, out: str) -> dict | None:
    cmd = [str(hconf.VENV_PY), str(hconf.BENCH / "score_report.py"),
           "--golden", golden, "--report", report, "--out", out,
           "--fetch-cache", str(hconf.FETCH_CACHE)]
    if tree:
        cmd += ["--tree", tree]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hconf.HARNESS),
                       timeout=900)
    p = hconf.HARNESS / out
    if r.returncode != 0 or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    vals = {"golden_count": 0, "golden_schema_ok": 0, "bun_first": 0, "measure_pairs": 0,
            "diversity_ok": 0, "llm_calls": 999, "fixtures_passed": 0, "baseline_queries": 0}
    reason = "-"

    goldens = hconf.load_golden()
    vals["golden_count"] = len(goldens)
    bad = [(g.get("id", "?"), schema_ok(g)) for g in goldens if schema_ok(g)]
    if bad:
        reason = f"golden_schema:{bad[0][0]}:{bad[0][1]}"
    else:
        vals["golden_schema_ok"] = 1

    ids = {g.get("id") for g in goldens}
    bun = next((g for g in goldens if g.get("id") == "bun-rust-port"), None)
    if bun:
        vals["bun_first"] = 1
    pairs = [g["id"] for g in goldens if g.get("measure_pair") is True]
    if len(pairs) == 2 and "bun-rust-port" in pairs:
        vals["measure_pairs"] = 2

    if vals["golden_schema_ok"] == 1 and goldens:
        d = diversity(goldens)
        if not d:
            vals["diversity_ok"] = 1
        elif reason == "-":
            reason = f"diversity:{d}"

    # scorer LLM scan (all python under bench/, the scorer plus any helpers it imports)
    hits = 0
    for p in sorted(hconf.BENCH.glob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace").lower()
        hits += sum(src.count(tok.lower()) for tok in LLM_TOKENS)
    vals["llm_calls"] = hits

    # fixture discrimination (law 5)
    fx = hconf.BENCH / "fixtures"
    if (hconf.BENCH / "score_report.py").exists() and (fx / "good_report.md").exists() \
            and (fx / "bad_report.md").exists() and (fx / "fixture_golden.txt").exists():
        gid = (fx / "fixture_golden.txt").read_text(encoding="utf-8").strip()
        gpath = f"bench/golden/{gid}.json"
        good = run_scorer(gpath, "bench/fixtures/good_report.md",
                          "bench/fixtures/good_tree.json" if (fx / "good_tree.json").exists() else None,
                          "no_read/evidence/fixture_good.scores.json")
        badx = run_scorer(gpath, "bench/fixtures/bad_report.md",
                          "bench/fixtures/bad_tree.json" if (fx / "bad_tree.json").exists() else None,
                          "no_read/evidence/fixture_bad.scores.json")
        if good and badx:
            up = all(good.get(k, 0) > badx.get(k, 0) for k in ("S1", "S2", "S4", "S5", "S6"))
            down = good.get("S3", 1) < badx.get("S3", 0)
            zero = good.get("llm_calls", 1) == 0 and badx.get("llm_calls", 1) == 0
            if up and down and zero:
                vals["fixtures_passed"] = 2
            elif reason == "-":
                reason = "fixtures_do_not_discriminate_on_all_6_metrics"
        elif reason == "-":
            reason = "score_report_failed_on_fixtures"
    elif reason == "-":
        reason = "score_report_or_fixtures_missing"

    # baseline provenance
    if hconf.BASELINE.exists():
        try:
            base = json.loads(hconf.BASELINE.read_text(encoding="utf-8"))
            q = base.get("queries", {})
            ok = 0
            for gid in sorted(ids):
                e = q.get(gid, {})
                rep = hconf.HARNESS / e.get("report", "missing")
                sc = hconf.HARNESS / e.get("scores", "missing")
                if rep.exists() and sc.exists() and all(f"S{i}_pct" in e for i in range(1, 7)):
                    ok += 1
            vals["baseline_queries"] = ok
            if ok < len(ids) and reason == "-":
                reason = f"baseline_covers_{ok}_of_{len(ids)}_goldens"
            if not all(f"S{i}_pct" in base.get("aggregate", {}) for i in range(1, 7)):
                vals["baseline_queries"] = 0
                if reason == "-":
                    reason = "baseline_aggregate_missing_S_pcts"
        except json.JSONDecodeError:
            reason = "baseline_unparseable"
    elif reason == "-":
        reason = "baseline_firecrawl_missing"

    ok_all = (vals["golden_count"] >= 5 and vals["golden_schema_ok"] == 1
              and vals["bun_first"] == 1 and vals["measure_pairs"] == 2
              and vals["diversity_ok"] == 1
              and vals["llm_calls"] == 0 and vals["fixtures_passed"] >= 2
              and vals["baseline_queries"] >= 5
              and vals["baseline_queries"] == vals["golden_count"])
    hconf.write_json(hconf.EVID / "s0_bench.json", {"stage": 0, **vals, "ok": 1 if ok_all else 0})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} ok={1 if ok_all else 0} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
