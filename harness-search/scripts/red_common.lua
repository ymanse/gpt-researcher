-- red_common.lua — RED integrity gate, shared by s1..s5_red.lua wrappers (law 3).
-- Proves the new tests exist, COLLECT, and FAIL: errors=0, collected>=1, passed=0,
-- failed>=1, test-file hashes + base_sha recorded. Also re-checks the bench freeze.
return function(stage)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local rel = "no_read/evidence/s" .. stage .. "_red.json"
  local blob = L.slurp(rel)
  if not blob then
    gralph.fail(rel .. " not found — write the RED tests in tests/search_quality/s" .. stage ..
      "/ then RUN: python scripts/pytest_evidence.py --stage " .. stage .. " --phase red")
    return
  end
  if not L.must(blob, '"stage":' .. stage .. ',',
      "evidence is for the wrong stage — regenerate with --stage " .. stage) then return end
  if not L.must(blob, '"phase":"red"', "evidence must come from a --phase red run") then return end
  if not L.must(blob, '"errors":0',
      "collection/import errors must be 0 — a test that won't import is not a valid RED") then return end
  local col = L.num(blob, '"collected":(%d+)')
  if not col or col == 0 then
    gralph.fail("s" .. stage .. ": 0 tests collected — write real pytest tests in tests/search_quality/s" .. stage .. "/")
    return
  end
  if not L.must(blob, '"passed":0',
      "a fresh RED test must NOT pass — do not implement before the RED gate") then return end
  local failed = L.num(blob, '"failed":(%d+)')
  if not failed or failed == 0 then
    gralph.fail("s" .. stage .. ": tests did not FAIL — RED means the suite fails because the implementation is missing")
    return
  end
  if not L.must(blob, '"test_files":{"',
      "evidence must record test-file sha256 hashes (pytest_evidence.py emits them) — the impl gate compares against these") then return end
  if not L.must(blob, '"base_sha":"',
      "evidence must record base_sha (HEAD at RED time) — review_diff.py diffs against it") then return end
  if not L.check_frozen() then return end
end
