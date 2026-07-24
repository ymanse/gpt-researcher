-- s2_smoke.lua — Stage 2 live smoke: learnings compression actually relaxed.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(2, function(blob, L)
  local lc = L.num(blob, '"learnings_count":(%d+)')
  if not lc or lc < 6 then
    gralph.fail("learnings_count<6 in the live deep_research (baseline cap was 3) — the config " ..
      "DEEP_RESEARCH_LEARNINGS default 8 / max_tokens 2500 is not effective at runtime; " ..
      "check the TIERA_EVIDENCE stage=2 line is emitted from process_research_results")
    return false
  end
  return true
end)
