-- dedup-audit gate — audits the GATES, not the build. DONE on pass.
-- audit_check.py is re-run IN-GATE against this profile: law-6 try matrix over every
-- dedup.yaml node, law-8 three-way token sync (gate <-> guidance <-> DEDUP-HARNESS.md),
-- law-10 regenerator idempotence, the score-tamper git check, and store_untampered.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local out = L.popen("python scripts/audit_check.py --profile dedup.yaml", "audit_check.py")
if not out then return end
if not out:find("ok=1", 1, true) then
  local why = out:match("reason=(.+)$") or "no verdict"
  gralph.fail("harness audit FAILED: " .. why .. " — fix what audit_check.py reports. " ..
    "The harness itself is the one thing you may change here, and only when the harness is " ..
    "genuinely wrong: gate + guidance + DEDUP-HARNESS.md move in the SAME commit (law 8)")
  return
end
if L.num(out, "nodes_tried=(%d+)") == 0 then
  gralph.fail("harness audit: nodes_tried=0 — an empty try matrix is not a pass. Probe every node " ..
    "with scripts/try_probe.py --profile dedup.yaml (pass AND fail evidence)")
  return
end
