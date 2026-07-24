-- scripts/lib.lua — shared gate helpers, loaded via dofile at the top of each gate.
-- gopher-lua has no JSON lib, so gates validate tool-emitted evidence files by literal
-- substring after stripping whitespace. Principle: FAIL CLOSED — a missing file or a
-- missing/negative field is a FAIL, never a silent pass. (If dofile is unavailable in
-- your gralph build, `gralph validate` flags it immediately — inline these two fns then.)
local L = {}

-- read a file under the profile dir, whitespace-stripped so `"k": 0` matches `"k":0`.
function L.slurp(rel)
  local f = io.open(gralph.profile_dir .. "/" .. rel, "r")
  if not f then return nil end
  local s = f:read("*a"); f:close()
  return (s:gsub("%s+", ""))
end

-- extract a number from a blob (evidence file OR a scanner's stdout), nil when absent.
-- NEVER write `tonumber(blob:match(pat))`: in Lua `tonumber(nil)` RAISES, so a missing field
-- kills the gate with a stack trace instead of a prescriptive fail — and failures.json then
-- carries that trace into the next session, which cannot act on it. (Proven in production:
-- dropping "collected" from an impl report yielded `SCRIPT ERROR: bad argument #1 to tonumber`.)
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

return L
