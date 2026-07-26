-- measure_common.lua — live-measure gate skeleton, shared by s1..s5_measure.lua.
-- Usage:  dofile(...)(stage, next_node, function(blob, L) ...threshold checks... end)
-- Verifies: evidence is for THIS stage and THIS bench_round (stale-evidence guard),
-- container was force-recreated with /health 200 (recorded by measure.py, the file's
-- only writer), per-stage thresholds, bench freeze, [sq][sN] commit. Then routes:
-- in a benchmark-refit round (store bench_refit == "sN") back to s6-benchmark,
-- otherwise to next_node. Stages 1-4 have 2 successors -> explicit gralph.route().
return function(stage, next_node, checks)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local rel = "no_read/evidence/s" .. stage .. "_measure.json"
  local blob = L.slurp(rel)
  if not blob then
    gralph.fail(rel .. " not found — RUN: python scripts/measure.py --stage " .. stage); return
  end
  -- "stage" is the alphabetically-last key for some stage schemas (sort_keys=True in
  -- hconf.write_json), so it serializes as `"stage":N}` (no trailing comma) instead of
  -- `"stage":N,`. Accept either terminator — this only widens the match, the required
  -- value is still the exact stage number.
  if not (blob:find('"stage":' .. stage .. ',', 1, true) or blob:find('"stage":' .. stage .. '}', 1, true)) then
    gralph.fail('gate FAIL: missing/!= "stage":' .. stage .. ' — evidence is for the wrong stage — rerun measure.py --stage ' .. stage)
    return
  end
  local br = tonumber(gralph.store.get("bench_round")) or 0
  local ebr = L.num(blob, '"bench_round":(%d+)')
  if ebr ~= br then
    gralph.fail("measure evidence bench_round=" .. tostring(ebr) .. " != store bench_round=" .. br ..
      " — STALE evidence from an earlier round; re-run python scripts/measure.py --stage " .. stage)
    return
  end
  if not L.must(blob, '"recreated":true',
      "the container must be force-recreated so the bind-mounted source is live — rerun measure.py") then return end
  if not L.must(blob, '"health":200',
      "/health did not return 200 after recreate — check docker logs gptr-mcp-server") then return end

  -- Freshness w.r.t. the CODE, recomputed in-gate: numbers must describe the
  -- implementation currently on disk. Editing the implementation after measuring (or
  -- re-scoring a cached tree the old code produced) can never reach the gate.
  local fpout = L.popen("python scripts/code_fp.py", "code_fp.py")
  if not fpout then return end
  local fp = fpout:match("code_fp=(%x+)")
  if not fp then
    gralph.fail("code_fp.py printed no fingerprint — run `python scripts/code_fp.py` and fix what it reports")
    return
  end
  if not blob:find('"code_fp":"' .. fp .. '"', 1, true) then
    gralph.fail("s" .. stage .. " measure evidence was produced by DIFFERENT implementation bytes than " ..
      "the ones on disk now (current code_fp=" .. fp .. ") — the implementation changed after the " ..
      "measurement, so the numbers describe code that no longer exists. Re-run: python scripts/measure.py --stage " ..
      stage .. " (it re-runs the live tree queries because the fingerprint moved)")
    return
  end

  if not checks(blob, L) then return end
  if not L.check_frozen() then return end
  if not L.check_commit(stage) then return end

  local refit = gralph.store.get("bench_refit")
  if refit == ("s" .. stage) then
    gralph.store.set("bench_refit", "")
    gralph.route("s6-benchmark")
  else
    gralph.route(next_node)
  end
end
