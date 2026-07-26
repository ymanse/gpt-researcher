-- impl_common.lua — GREEN anti-tamper gate, shared by s1..s5_impl.lua wrappers.
-- The decisive check is recomputed IN-GATE (never trusted from the agent): verify_impl.py
-- re-hashes the RED test files from disk, re-runs the stage suite AND the full cumulative
-- tests/search_quality suite, ruffs the changed files, re-checks the bench freeze, and
-- prints one machine line. A forged evidence file cannot survive this.
return function(stage)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local out = L.popen("python scripts/verify_impl.py --stage " .. stage, "verify_impl.py")
  if not out then return end
  local function n(k) return L.num(out, k .. "=(%d+)") end

  if n("red_evidence") ~= 1 then
    gralph.fail("s" .. stage .. ": RED evidence missing/unreadable — the red stage must have produced s" ..
      stage .. "_red.json; do not delete evidence")
    return
  end
  if n("hash_match") ~= 1 then
    gralph.fail("s" .. stage .. ": RED test files were MODIFIED since the red gate (sha256 mismatch) — " ..
      "revert the test files byte-for-byte; change only the implementation")
    return
  end
  if n("red_collected_ok") ~= 1 then
    gralph.fail("s" .. stage .. ": collected test count shrank below the RED count — tests were deleted, restore them")
    return
  end
  local sc = n("stage_collected")
  if not sc or sc == 0 then
    gralph.fail("s" .. stage .. ": 0 tests collected in tests/search_quality/s" .. stage .. " — a hollow zero is not GREEN")
    return
  end
  if n("stage_errors") ~= 0 then gralph.fail("s" .. stage .. ": collection/import errors in the stage suite — fix them"); return end
  if n("stage_failed") ~= 0 then gralph.fail("s" .. stage .. ": stage tests still failing — implement until GREEN, never weaken tests"); return end
  if n("stage_skipped") ~= 0 then gralph.fail("s" .. stage .. ": skipped tests detected — skipping is not passing; unskip them"); return end
  local fc = n("suite_collected")
  if not fc or fc < sc then
    gralph.fail("full tests/search_quality suite collected fewer tests than the stage suite — earlier stages' tests were removed")
    return
  end
  if n("suite_errors") ~= 0 then gralph.fail("full search_quality suite has collection errors — an earlier stage's tests no longer import"); return end
  if n("suite_failed") ~= 0 then gralph.fail("full search_quality suite not green — a previous stage REGRESSED; fix the implementation, never the old tests"); return end
  if n("suite_skipped") ~= 0 then gralph.fail("full search_quality suite has skipped tests — skipping is not passing"); return end
  local ruff = n("ruff_errors")
  if not ruff or ruff ~= 0 then
    gralph.fail("s" .. stage .. ": ruff_errors=" .. tostring(ruff) .. " — run `../venv/Scripts/python -m ruff check <changed files>` and fix every violation")
    return
  end
  if n("frozen_ok") ~= 1 then
    gralph.fail("bench/golden or baseline drifted from the s0 freeze — they are READ-ONLY; revert them")
    return
  end
  if n("review_addressed_ok") ~= 1 then
    gralph.fail("s" .. stage .. ": the last review round left blocking findings you have not acknowledged — fix each one, " ..
      "then write no_read/evidence/s" .. stage .. "_impl_ack.json = {\"addressed_findings\":[<every blocking id>]} and resubmit")
    return
  end
end
