-- s5_smoke.lua — Stage 5 live smoke: scope=on builds a brief, scope=off is regression-free.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(5, function(blob, L)
  if not L.must(blob, '"scope_on_brief_present":true',
      "scope=True live call produced no brief — the TIERA_EVIDENCE stage=5 brief line was not observed") then return false end
  local bl = L.num(blob, '"brief_len":(%d+)')
  if not bl or bl < 1 then
    gralph.fail("brief_len<1 — the brief object is empty; the 1-round clarification must produce a non-empty scope statement")
    return false
  end
  if not L.must(blob, '"scope_off_ok":true',
      "scope=False live call regressed — it must complete exactly like the current behavior, with NO brief generated") then return false end
  return true
end)
