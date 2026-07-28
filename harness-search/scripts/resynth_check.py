"""d0 verifier — run IN-GATE by d0_resynth.lua (and as the agent's own preflight).

    python scripts/resynth_check.py

Proves the offline re-synthesis runner is a faithful stand-in for a live run, because
every later dedup stage measures the runner's output instead of a $330-per-query live one.
If it is not faithful, the whole loop optimises a fiction (law 4/5), so this is the
decisive gate of the harness — not "does scripts/resynth.py exist".

Recomputed here, never taken from a claim:

  corpus_queries        captured goldens in no_read/dedup/corpus (must be the full set)
  corpus_fresh          the corpus was captured by the implementation bytes on disk now
  bytes_identical       re-synthesising each captured tree reproduces that run's OWN
                        report.md byte for byte. Byte identity is reachable because the
                        whole assembly path is deterministic — create_chat_completion
                        appears twice in tree_research.py and both sites are upstream of
                        the roll-up — so anything less means the sidecar lost state
                        (node order, sources, learnings, read_docs) and the offline
                        numbers would drift from live in a way nobody can see.
  netblocked            outbound socket attempts during those re-syntheses, counted by
                        scripts/netguard/sitecustomize.py from inside the probed process.
                        Must be 0: the roll-up may not retrieve.
  shares_assembly       resynth.py CALLS the skill's assembly instead of re-implementing
                        it. A private copy would pass fidelity today and silently diverge
                        the moment s9 changes the real one (law 5 cross-check).
  read_docs_min         smallest scraped-document count across the corpus (positive
                        binding: an empty read_docs makes a passing fidelity hollow)
  suite_*               tests/search_quality + tests/tier_a re-run — d0 refactors shared
                        implementation code, so the existing contracts must stay green
  frozen_ok             bench/golden + baseline + scorer unchanged since the s0 freeze

Prints ONE machine line; writes no_read/evidence/d0_resynth.json. Idempotent: same corpus
and same code -> same bytes, so it is also the law-10 regenerator for d0 evidence.
"""
from __future__ import annotations

import json
import os
import subprocess

import code_fp
import hconf

DEDUP = hconf.HARNESS / "no_read" / "dedup"
CORPUS = DEDUP / "corpus"
FIDELITY = DEDUP / "fidelity"
NETGUARD = hconf.HARNESS / "scripts" / "netguard"
RESYNTH = hconf.HARNESS / "scripts" / "resynth.py"
TIER_A = hconf.REPO / "tests" / "tier_a"

# a private copy of the assembly would pass fidelity today and diverge the moment s9
# edits the real one — these names may only live in the implementation
FORBIDDEN_IN_RUNNER = ("_attribute_citations", "verify_rollup", "_prune_ungrounded_markers")


def run_one(gid: str, netlog) -> tuple[bool, str]:
    """Re-synthesise one captured tree offline; True when the bytes match the live report."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(NETGUARD) + os.pathsep + env.get("PYTHONPATH", "")
    env["SQ_NETBLOCK_LOG"] = str(netlog)
    env["PYTHONUTF8"] = "1"
    r = subprocess.run(
        [str(hconf.VENV_PY), str(RESYNTH),
         "--resynth", str(CORPUS / f"{gid}.resynth.json"),
         "--tree", str(CORPUS / f"{gid}.tree.json"),
         "--out", str(FIDELITY), "--as", gid],
        capture_output=True, text=True, errors="replace", env=env,
        cwd=str(hconf.HARNESS), timeout=3600,
    )
    out = FIDELITY / f"{gid}.report.md"
    if r.returncode != 0:
        return False, f"{gid}: resynth.py exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}"
    if not out.exists():
        return False, f"{gid}: resynth.py wrote no {gid}.report.md into {FIDELITY.name}/"
    want = hconf.sha256(CORPUS / f"{gid}.report.md")
    got = hconf.sha256(out)
    if want != got:
        a = (CORPUS / f"{gid}.report.md").read_text(encoding="utf-8", errors="replace")
        b = out.read_text(encoding="utf-8", errors="replace")
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return False, (f"{gid}: report differs from the live run at char {first} "
                       f"(live {len(a)}c, resynth {len(b)}c) — the sidecar is missing state; "
                       f"live[{first}:{first + 60}]={a[first:first + 60]!r} "
                       f"resynth[{first}:{first + 60}]={b[first:first + 60]!r}")
    return True, ""


def main() -> int:
    vals = {"corpus_queries": 0, "golden_count": 0, "corpus_fresh": 0, "bytes_identical": 0,
            "netblocked": 999, "shares_assembly": 0, "read_docs_min": 0,
            "suite_collected": 0, "suite_failed": 0, "suite_errors": 1, "suite_skipped": 0,
            "frozen_ok": 0}
    reason = "-"
    errors: list[str] = []
    fp = code_fp.fingerprint()

    goldens = [g["id"] for g in hconf.load_golden()]
    vals["golden_count"] = len(goldens)

    man = {}
    try:
        man = json.loads((CORPUS / "corpus.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        reason = "corpus_missing_run_capture_py"

    rows = man.get("queries") or []
    vals["corpus_queries"] = len(rows)
    vals["corpus_fresh"] = 1 if man.get("code_fp") == fp else 0
    if rows and not vals["corpus_fresh"] and reason == "-":
        reason = (f"corpus_code_fp={man.get('code_fp')}_!=_current={fp}"
                  "_the_reference_reports_came_from_code_that_no_longer_exists")
    vals["read_docs_min"] = min((int(r.get("read_docs_urls", 0)) for r in rows), default=0)

    # law 5: the runner must delegate, not duplicate
    if RESYNTH.exists():
        src = RESYNTH.read_text(encoding="utf-8", errors="replace")
        impl = (hconf.REPO / "gpt_researcher" / "skills" / "tree_research.py").read_text(
            encoding="utf-8", errors="replace")
        dup = [n for n in FORBIDDEN_IN_RUNNER if n in src]
        shares = "assemble_report" in src and "def assemble_report" in impl
        vals["shares_assembly"] = 1 if shares and not dup else 0
        if not shares and reason == "-":
            reason = ("resynth.py_must_call_the_skill's_assemble_report;"
                      "_extract_the_assembly_out_of_run()_into_that_method")
        elif dup and reason == "-":
            reason = f"resynth.py_re-implements_the_assembly:{','.join(dup)}"
    elif reason == "-":
        reason = "scripts/resynth.py_missing"

    # fidelity: only meaningful once the corpus is complete and fresh
    if rows and vals["corpus_fresh"] and RESYNTH.exists():
        FIDELITY.mkdir(parents=True, exist_ok=True)
        netlog = DEDUP / "netblock.log"
        netlog.unlink(missing_ok=True)
        same = 0
        for r in rows:
            ok, err = run_one(r["id"], netlog)
            if ok:
                same += 1
            else:
                errors.append(err)
        vals["bytes_identical"] = same
        vals["netblocked"] = (len(netlog.read_text(encoding="utf-8", errors="replace")
                                  .splitlines()) if netlog.exists() else 0)
        if errors and reason == "-":
            reason = errors[0][:200]
        elif vals["netblocked"] and reason == "-":
            reason = f"offline_resynth_attempted_{vals['netblocked']}_outbound_connections"

    # the d0 refactor touches shared implementation code — existing contracts stay green
    counts = hconf.run_pytest(hconf.TESTS, hconf.EVID / "d0_suite_sq.xml")
    tier = (hconf.run_pytest(TIER_A, hconf.EVID / "d0_suite_tier_a.xml")
            if TIER_A.exists() else {"collected": 0, "errors": 0, "passed": 0,
                                     "failed": 0, "skipped": 0})
    for k in ("collected", "failed", "errors", "skipped"):
        vals[f"suite_{k}"] = counts[k] + tier[k]

    r = subprocess.run([str(hconf.VENV_PY), "scripts/check_frozen.py"],
                       capture_output=True, text=True, cwd=str(hconf.HARNESS), timeout=120)
    vals["frozen_ok"] = 1 if "frozen_ok=1" in (r.stdout or "") else 0

    ok_all = (
        vals["corpus_queries"] == vals["golden_count"] and vals["golden_count"] >= 5
        and vals["corpus_fresh"] == 1
        and vals["bytes_identical"] == vals["corpus_queries"]
        and vals["netblocked"] == 0 and vals["shares_assembly"] == 1
        and vals["read_docs_min"] >= 10
        and vals["suite_collected"] >= 83 and vals["suite_failed"] == 0
        and vals["suite_errors"] == 0 and vals["suite_skipped"] == 0
        and vals["frozen_ok"] == 1
    )
    hconf.write_json(hconf.EVID / "d0_resynth.json",
                     {"stage": "d0", "phase": "resynth", **vals, "code_fp": fp,
                      "errors": errors, "ok": 1 if ok_all else 0})
    line = " ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{line} ok={1 if ok_all else 0} reason={reason}")
    for e in errors[:5]:
        print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
