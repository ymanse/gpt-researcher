-- s4_smoke.lua — Stage 4 live smoke: academic lane on Firecrawl research + citers expansion.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(4, function(blob, L)
  local pc = L.num(blob, '"papers_count":(%d+)')
  if not pc or pc < 3 then
    gralph.fail("papers_count<3 in the live academic query — the research adapter is not returning papers; " ..
      "check the TIERA_EVIDENCE stage=4 line and the academic routing")
    return false
  end
  local w = L.num(blob, '"papers_in_window":(%d+)')
  if not w or w < 1 then
    gralph.fail("papers_in_window<1 — no paper inside the from/to recency window; the window filter is not applied or too narrow")
    return false
  end
  local c = L.num(blob, '"from_citers_count":(%d+)')
  if not c or c < 1 then
    gralph.fail("from_citers_count<1 — citers expansion added nothing; related-papers mode=citers must contribute at least one paper")
    return false
  end
  return true
end)
