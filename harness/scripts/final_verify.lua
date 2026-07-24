-- final_verify.lua — capstone gate (single successor: DONE). The decisive checks are
-- recomputed IN-GATE by final_verify.py: full tests/tier_a suite green, all 18 evidence
-- files present and stage-tagged, all 6 [tier-a][stageN] commits present (gptr-mcp repo
-- for stages 5-6), tracked trees clean.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local p = io.popen("python scripts/final_verify.py")
if not p then gralph.fail("could not launch final_verify.py"); return end
local out = p:read("*a"); p:close()
if not out or out == "" then
  gralph.fail("final_verify.py produced no output — run `python scripts/final_verify.py` manually and fix what it reports")
  return
end
out = out:gsub("%s+", " ")
local function n(k) return L.num(out, k .. "=(%d+)") end

local col = n("suite_collected")
if not col or col == 0 then gralph.fail("full tier_a suite collected 0 tests — a hollow zero cannot finish the build"); return end
if n("suite_errors") ~= 0 then gralph.fail("full tier_a suite has collection errors — some stage's tests no longer import"); return end
if n("suite_failed") ~= 0 then gralph.fail("full tier_a suite not green — fix the implementation, never the tests"); return end
if n("suite_skipped") ~= 0 then gralph.fail("full tier_a suite has skipped tests — skipping is not passing"); return end
if n("evidence_ok") ~= 1 then
  local why = out:match("evidence_reason=([^|]+)") or "an evidence file is missing or mistagged"
  gralph.fail("evidence incomplete: " .. why .. " — every stage needs red/green/smoke evidence produced by the scripts")
  return
end
if n("commits_ok") ~= 1 then
  local why = out:match("commit_reason=([^|]+)") or "a stage commit is missing or a tree is dirty"
  gralph.fail("commit check failed: " .. why)
  return
end
