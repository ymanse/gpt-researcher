-- s6-benchmark gate — the final contest, and the router back to the weakest stage.
-- Evidence comes ONLY from benchmark.py (all 5 golden queries, round-cached, scored by
-- the frozen scorer against the frozen baseline). queries_scored < golden_count is a
-- hollow zero (law 4). On any metric below baseline: route to the owning stage's impl
-- (S1->s2, S2->s1, S3->s3, S4->s4, S5->s4, S6->s5), max 3 rounds.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local blob = L.slurp("no_read/evidence/benchmark.json")
if not blob then
  gralph.fail("no_read/evidence/benchmark.json not found — RUN: python scripts/benchmark.py"); return
end
local br = tonumber(gralph.store.get("bench_round")) or 0
local ebr = L.num(blob, '"bench_round":(%d+)')
if ebr ~= br then
  gralph.fail("benchmark evidence bench_round=" .. tostring(ebr) .. " != store bench_round=" .. br ..
    " — STALE; re-run python scripts/benchmark.py")
  return
end
if not L.must(blob, '"recreated":true', "container must be force-recreated — rerun benchmark.py") then return end
if not L.must(blob, '"health":200', "/health != 200 — check docker logs gptr-mcp-server") then return end
-- the final contest must describe the implementation on disk right now (see measure_common)
local fpout = L.popen("python scripts/code_fp.py", "code_fp.py")
if not fpout then return end
local fp = fpout:match("code_fp=(%x+)")
if not fp then gralph.fail("code_fp.py printed no fingerprint"); return end
if not blob:find('"code_fp":"' .. fp .. '"', 1, true) then
  gralph.fail("benchmark evidence was produced by DIFFERENT implementation bytes than the ones on disk " ..
    "now (current code_fp=" .. fp .. ") — re-run: python scripts/benchmark.py")
  return
end
local qs = L.num(blob, '"queries_scored":(%d+)')
local gc = L.num(blob, '"golden_count":(%d+)')
if not qs or not gc or gc < 5 or qs < gc then
  gralph.fail("queries_scored=" .. tostring(qs) .. " of golden_count=" .. tostring(gc) ..
    " — a partial benchmark is a hollow zero; fix the per-query errors[] in benchmark.json and re-run " ..
    "python scripts/benchmark.py (completed queries are round-cached, only failures re-run)")
  return
end
if not L.check_frozen() then return end

if blob:find('"all_pass":1', 1, true) then
  gralph.route("harness-audit")
  return
end
local weakest = blob:match('"weakest_metric":"(S%d)"')
if not weakest then
  gralph.fail('all_pass=0 but no "weakest_metric":"S<k>" recorded — re-run benchmark.py (it computes the largest shortfall)')
  return
end
if br >= 3 then
  gralph.fail("benchmark rounds exhausted (3) and " .. weakest .. " is still not beating the baseline — " ..
    "human review needed: see no_read/evidence/benchmark.json per_metric")
  return
end
local owner = { S1 = "s2", S2 = "s1", S3 = "s3", S4 = "s4", S5 = "s4", S6 = "s5" }
local target = owner[weakest]
gralph.store.set("bench_round", br + 1)
gralph.store.set("bench_refit", target)
gralph.route(target .. "-impl")
