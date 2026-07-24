-- s3_smoke.lua — Stage 3 live smoke: citation verification pass grounded >= 70%.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(3, function(blob, L)
  local tc = L.num(blob, '"total_claims":(%d+)')
  if not tc or tc < 1 then
    gralph.fail("total_claims<1 — the citation-verification pass did not run on the live call; " ..
      "check the TIERA_EVIDENCE stage=3 line is emitted after the pass")
    return false
  end
  if not L.must(blob, '"unverified":',
      "the unverified count must be reported — an evidence file that hides it proves nothing") then return false end
  local gr = L.num(blob, '"grounded_ratio_pct":(%d+)')
  if not gr or gr < 70 then
    gralph.fail("grounded_ratio_pct<70 — fewer than 70% of claims resolve to their cited source; " ..
      "improve quote matching (normalized containment) or citation capture, never loosen the check")
    return false
  end
  return true
end)
