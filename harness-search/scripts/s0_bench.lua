-- s0-bench gate — the benchmark harness everything else depends on.
-- Decisive checks are recomputed IN-GATE by bench_selfcheck.py (never trusted from the
-- agent): golden schema, fixture discrimination (law 5), LLM-free scorer (law 2),
-- baseline provenance. Then the freeze manifest and the [sq][s0] commit are verified.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local out = L.popen("python scripts/bench_selfcheck.py", "bench_selfcheck.py")
if not out then return end
local function n(k) return L.num(out, k .. "=(%d+)") end

local gc = n("golden_count")
if not gc or gc < 5 then
  gralph.fail("golden_count=" .. tostring(gc) .. " < 5 — author at least 5 golden sets under bench/golden/ "
    .. "(id, query, required_primary_domains, facts, traps, contested, coverage_areas)")
  return
end
if n("golden_schema_ok") ~= 1 then
  gralph.fail("a golden file is schema-incomplete or has a non-compiling regex — run "
    .. "`python scripts/bench_selfcheck.py` and fix the reason= it prints")
  return
end
if n("bun_first") ~= 1 then
  gralph.fail("bench/golden must contain id \"bun-rust-port\" seeded from "
    .. "D:/dev_ext/gptr-mcp/outputs/how-did-bun-port-*.tree.json (meta.query verbatim)")
  return
end
if n("measure_pairs") ~= 2 then
  gralph.fail("exactly 2 goldens must set \"measure_pair\":true and one must be bun-rust-port "
    .. "— s2-s5 measures run only that pair")
  return
end
if n("diversity_ok") ~= 1 then
  local dr = out:match("reason=diversity:([^ ]+)") or "see bench_selfcheck reason"
  gralph.fail("golden-set diversity floor not met (" .. dr .. ") — per golden: facts>=5, traps>=3, "
    .. "coverage_areas>=4, required_primary_domains>=2, contested>=1, category from the spec taxonomy; "
    .. "across the set: >=4 distinct categories, >=2 dated goldens (a fact matching 202[4-9]), "
    .. "measure_pair categories must differ with >=1 dated member, >=6 distinct primary domains. "
    .. "Timeless best-practice queries make the trap metric vacuous — see spec/search-quality.md")
  return
end
if n("llm_calls") ~= 0 then
  gralph.fail("llm_calls != 0 — the scorer bench/*.py contains an LLM SDK import/call; the "
    .. "scorer must be fully deterministic (law 2). Remove every model call.")
  return
end
if n("fixtures_passed") ~= 2 then
  gralph.fail("fixtures_passed != 2 — score_report.py must score bench/fixtures/{good,bad}_report.md "
    .. "and good must beat bad STRICTLY on S1,S2,S4,S5,S6 and be strictly lower on S3 (law 5)")
  return
end
local bq = n("baseline_queries")
if not bq or bq < 5 or bq ~= gc then
  gralph.fail("baseline_queries=" .. tostring(bq) .. " — bench/baseline_firecrawl.json must score EVERY "
    .. "golden id with S1_pct..S6_pct + report/scores provenance under bench/baseline_runs/")
  return
end
-- freeze + commit: manifest must exist and match, and [sq][s0] must be committed clean.
if not L.check_frozen() then return end
if not L.check_commit(0) then return end
