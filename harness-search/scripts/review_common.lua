-- review_common.lua — adversarial review verdict gate, shared by s1..s5_review.lua.
-- The review agent (separate lane: gets ONLY git diff + completion conditions) writes
-- no_read/evidence/sN_review.json. This gate verifies the verdict is about the CURRENT
-- code (head_sha recomputed in-gate) and ROUTES: blocking findings -> back to sN-impl
-- (max 3 rounds, counted in the store on the success path), clean -> sN-measure.
-- NOTE deliberate law-2 deviation: the verdict itself is LLM judgment (user-specified
-- reviewer lane); what stays deterministic is freshness, shape, and the round cap.
return function(stage)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local rel = "no_read/evidence/s" .. stage .. "_review.json"
  local blob = L.slurp(rel)
  if not blob then
    gralph.fail(rel .. " not found — run `python scripts/review_diff.py --stage " .. stage ..
      "`, review the diff against spec/search-quality.md s" .. stage .. " completion conditions " ..
      "(assume the code is WRONG and say why), then write the review JSON")
    return
  end
  if not L.must(blob, '"stage":' .. stage .. ',',
      "review evidence is for the wrong stage") then return end

  local head = L.popen("python scripts/head_sha.py", "head_sha.py")
  if not head then return end
  head = head:gsub("%s+", "")
  if not blob:find('"head_sha":"' .. head .. '"', 1, true) then
    gralph.fail("review.json head_sha != current HEAD (" .. head .. ") — the review is about older code; " ..
      "re-run review_diff.py and re-review the CURRENT diff")
    return
  end
  if not L.must(blob, '"findings":',
      'report a "findings" array (may be empty) — each {"id","severity":"blocking"|"minor","file","why"}') then return end
  local bc = L.num(blob, '"blocking_count":(%d+)')
  if not bc then
    gralph.fail('report a numeric "blocking_count" (count of severity:"blocking" findings)')
    return
  end

  if bc > 0 then
    local key = "rev:s" .. stage
    local rounds = tonumber(gralph.store.get(key)) or 0
    if rounds >= 3 then
      gralph.fail("s" .. stage .. ": 3 review rounds exhausted and blocking findings persist — " ..
        "a human must inspect no_read/evidence/s" .. stage .. "_review.json")
      return
    end
    gralph.store.set(key, rounds + 1)
    gralph.route("s" .. stage .. "-impl")
  else
    gralph.route("s" .. stage .. "-measure")
  end
end
