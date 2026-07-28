-- review_common.lua — adversarial review verdict gate, shared by s1..s5_review.lua.
-- The review agent (separate lane: gets ONLY git diff + completion conditions) writes
-- no_read/evidence/sN_review.json. This gate verifies the verdict is about the CURRENT
-- code (head_sha recomputed in-gate) and ROUTES: blocking findings -> back to sN-impl
-- (max 3 rounds, counted in the store on the success path), clean -> sN-measure.
-- NOTE deliberate law-2 deviation: the verdict itself is LLM judgment (user-specified
-- reviewer lane); what stays deterministic is freshness, shape, and the round cap.
-- next_node defaults to sN-measure (search-quality); dedup.yaml passes "d1-offline".
return function(stage, next_node, instance)
  next_node = next_node or ("s" .. stage .. "-measure")
  instance = instance or "search-quality"   -- journal dir: .gralph/<instance>/
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
    -- Round count comes from gralph's OWN append-only journal, never from the store:
    -- a blocked agent can rewrite .gralph/store.json (measured 2026-07-27, an agent reset
    -- rev:s2 3->0 and rewound the cursor after hitting this cap). Extra rounds must be
    -- granted explicitly by a human in no_read/audit/grants.json, where harness-audit
    -- surfaces them.
    local out = L.popen("python scripts/loop_audit.py --instance " .. instance, "loop_audit.py")
    if not out then return end
    local spent = L.num(out, "rev_journal_s" .. stage .. "=(%d+)")
    local granted = L.num(out, "grant_rev_s" .. stage .. "=(%d+)") or 0
    if not spent then
      gralph.fail("loop_audit.py did not report rev_journal_s" .. stage .. " — run it manually and fix what it reports")
      return
    end
    local effective = spent - granted
    if effective >= 3 then
      gralph.fail("s" .. stage .. ": " .. effective .. " blocking review rounds already spent (counted from " ..
        "journal.jsonl, " .. granted .. " human-granted) — this cap hands the decision to a HUMAN. " ..
        "Do NOT edit .gralph/ state, do not rewind the cursor, do not reset a counter: that is tampering " ..
        "and the journal records it anyway. STOP and let a human read no_read/evidence/s" .. stage .. "_review.json.")
      return
    end
    gralph.store.set("rev:s" .. stage, effective + 1)   -- informational mirror only
    gralph.route("s" .. stage .. "-impl")
  else
    gralph.route(next_node)
  end
end
