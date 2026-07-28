-- scripts/lib.lua — shared gate helpers, loaded via dofile at the top of each gate.
-- gopher-lua has no JSON lib, so gates validate tool-emitted evidence files by literal
-- substring after stripping whitespace. Principle: FAIL CLOSED — a missing file or a
-- missing/negative field is a FAIL, never a silent pass.
local L = {}

-- read a file under the profile dir, whitespace-stripped so `"k": 0` matches `"k":0`.
function L.slurp(rel)
  local f = io.open(gralph.profile_dir .. "/" .. rel, "r")
  if not f then return nil end
  local s = f:read("*a"); f:close()
  return (s:gsub("%s+", ""))
end

-- extract a number from a blob (evidence file OR a scanner's stdout), nil when absent.
-- NEVER write `tonumber(blob:match(pat))`: in Lua `tonumber(nil)` RAISES, so a missing
-- field would kill the gate with a stack trace instead of a prescriptive fail.
function L.num(blob, pat)
  if not blob then return nil end
  local m = blob:match(pat)
  return m and tonumber(m) or nil
end

-- require literal token present in blob; on absence set a prescriptive fail reason.
-- returns false so the caller can `return` (gralph keeps the FIRST fail reason).
function L.must(blob, token, why)
  if not blob then gralph.fail("evidence file missing — produce it then resubmit"); return false end
  if not blob:find(token, 1, true) then
    gralph.fail("gate FAIL: missing/!= " .. token .. (why and (" — " .. why) or ""))
    return false
  end
  return true
end

-- run a harness script in-gate (cmd-safe: relative forward-slash path, no quotes) and
-- return its stdout squashed to single spaces, or nil after a prescriptive fail.
-- Retries once on empty output: under load a python interpreter start occasionally
-- yields nothing, and treating that as a verdict failed three real gates (head_sha.py
-- twice, code_fp.py once) on work that was actually fine. A genuinely broken command
-- fails both attempts, so this loses no strictness.
function L.popen(cmd, what)
  for attempt = 1, 2 do
    local p = io.popen(cmd)
    if p then
      local out = p:read("*a"); p:close()
      if out and out ~= "" then return (out:gsub("%s+", " ")) end
    end
    if attempt == 2 then
      gralph.fail(what .. " produced no output on two attempts — run `" .. cmd ..
        "` manually and fix what it reports")
      return nil
    end
  end
end

-- bench read-only law: recompute golden/baseline hashes against bench/manifest.sha256.
function L.check_frozen()
  local out = L.popen("python scripts/check_frozen.py", "check_frozen.py")
  if not out then return false end
  if not out:find("frozen_ok=1", 1, true) then
    local why = out:match("reason=([^ ]+)") or "unknown"
    gralph.fail("bench/golden or baseline_firecrawl.json drifted from the s0 freeze (" .. why ..
      ") — they are READ-ONLY after s0; revert them (git checkout), never re-freeze")
    return false
  end
  return true
end

-- commit convention: both repos on feature/search-quality, clean tracked trees, and a
-- [sq][sN] commit in gpt-researcher.
-- prefix defaults to "s" (search-quality lanes); dedup.yaml passes "d" -> [sq][d0].
function L.check_commit(stage, prefix)
  prefix = prefix or "s"
  local out = L.popen("python scripts/check_commit.py --stage " .. stage ..
    " --prefix " .. prefix, "check_commit.py")
  if not out then return false end
  if not out:find("commits_ok=1", 1, true) then
    local why = out:match("reason=(.+)$") or "no verdict"
    gralph.fail("[sq][" .. prefix .. stage .. "] commit gate: " .. why ..
      " — commit ALL tracked changes on feature/search-quality with that message prefix, then resubmit")
    return false
  end
  return true
end

return L
