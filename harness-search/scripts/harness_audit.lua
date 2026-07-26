-- harness-audit gate — audits the GATES, not the build. Decisive checks recomputed
-- IN-GATE by audit_check.py: law 6 (gralph-try pass/fail reports for every node),
-- score-tamper git log, freeze manifest, law 10 regen idempotence, law 8 3-way sync.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local out = L.popen("python scripts/audit_check.py", "audit_check.py")
if not out then return end
local function n(k) return L.num(out, k .. "=(%d+)") end
local reason = out:match("reason=([^ ]+)") or "unknown"

if n("try_ok") ~= 1 then
  gralph.fail("law 6 audit failed (" .. reason .. ") — for EVERY node produce no_read/audit/try/<node>_pass.json " ..
    "and <node>_fail.json via `python scripts/try_probe.py --node <node> --expect pass|fail --report <file>` " ..
    "against known-good/known-bad probe evidence; back up real evidence by full path first and restore it " ..
    "with scripts/regen_evidence.py, never by hand")
  return
end
if n("git_ok") ~= 1 then
  gralph.fail("SCORE TAMPER detected (" .. reason .. ") — bench/golden/* or baseline_firecrawl.json was " ..
    "modified after the s0 freeze commit. This voids every downstream score. Revert the offending " ..
    "commit(s) so the frozen files match the s0 freeze, then re-run the audit")
  return
end
if n("frozen_ok") ~= 1 then
  gralph.fail("bench manifest mismatch (" .. reason .. ") — golden/baseline bytes drifted; revert them")
  return
end
if n("regen_ok") ~= 1 then
  gralph.fail("law 10 audit failed (" .. reason .. ") — a regenerator is not idempotent or evidence was " ..
    "hand-edited; every no_read/evidence file must be reproducible byte-identically by scripts/regen_evidence.py")
  return
end
if n("sync_ok") ~= 1 then
  gralph.fail("law 8 audit failed (" .. reason .. ") — a gate threshold token is missing from the profile " ..
    "guidance or HARNESS.md; edit gate+guidance+HARNESS.md together in the same commit")
  return
end
