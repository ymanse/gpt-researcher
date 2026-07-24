-- green_common.lua — GREEN anti-tamper gate, shared by s1..s6_green.lua wrappers.
-- The decisive check is recomputed IN-GATE (never trusted from the agent): verify_green.py
-- re-hashes the RED test files from disk, re-runs the stage suite AND the full cumulative
-- tests/tier_a suite, and prints one machine line. A forged evidence file cannot survive this.
-- os.execute/io.popen shell out via cmd.exe: relative forward-slash script path, no quotes.
return function(stage)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local p = io.popen("python scripts/verify_green.py --stage " .. stage)
  if not p then gralph.fail("could not launch verify_green.py — python must be on PATH"); return end
  local out = p:read("*a"); p:close()
  if not out or out == "" then
    gralph.fail("verify_green.py produced no output — run `python scripts/verify_green.py --stage " ..
      stage .. "` manually and fix what it reports")
    return
  end
  out = out:gsub("%s+", " ")
  local function n(k) return L.num(out, k .. "=(%d+)") end

  if n("red_evidence") ~= 1 then
    gralph.fail("stage " .. stage .. ": RED evidence missing/unreadable — the RED gate must have produced stage" ..
      stage .. "_red.json; do not delete evidence")
    return
  end
  if n("hash_match") ~= 1 then
    gralph.fail("stage " .. stage .. ": RED test files were MODIFIED since the RED gate (sha256 mismatch) — " ..
      "revert the test files byte-for-byte; change only the implementation")
    return
  end
  if n("red_collected_ok") ~= 1 then
    gralph.fail("stage " .. stage .. ": collected test count shrank below the RED count — tests were deleted, restore them")
    return
  end
  local sc = n("stage_collected")
  if not sc or sc == 0 then
    gralph.fail("stage " .. stage .. ": 0 tests collected in tests/tier_a/stage" .. stage .. " — a hollow zero is not GREEN")
    return
  end
  if n("stage_errors") ~= 0 then gralph.fail("stage " .. stage .. ": collection/import errors in the stage suite — fix them"); return end
  if n("stage_failed") ~= 0 then gralph.fail("stage " .. stage .. ": stage tests still failing — implement until GREEN, never weaken tests"); return end
  if n("stage_skipped") ~= 0 then gralph.fail("stage " .. stage .. ": skipped tests detected — skipping is not passing; unskip them"); return end
  local fc = n("suite_collected")
  if not fc or fc < sc then
    gralph.fail("full tests/tier_a suite collected fewer tests than the stage suite — earlier stages' tests were removed")
    return
  end
  if n("suite_errors") ~= 0 then gralph.fail("full tier_a suite has collection errors — an earlier stage's tests no longer import"); return end
  if n("suite_failed") ~= 0 then gralph.fail("full tier_a suite not green — a previous stage REGRESSED; fix the implementation, never the old tests"); return end
  if n("suite_skipped") ~= 0 then gralph.fail("full tier_a suite has skipped tests — skipping is not passing"); return end
end
