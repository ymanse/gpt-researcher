-- s1_smoke.lua — Stage 1 live smoke: firecrawl actually routed + clean-markdown bodies.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(1, function(blob, L)
  local fs = L.num(blob, '"firecrawl_sources":(%d+)')
  if not fs or fs < 1 then
    gralph.fail("no firecrawl-routed source observed in the live quick_search (firecrawl_sources<1) — " ..
      "the ROUTING_TABLE entry or the adapter is not effective at runtime; check the TIERA_EVIDENCE stage=1 log line is emitted")
    return false
  end
  local bl = L.num(blob, '"max_firecrawl_body_len":(%d+)')
  if not bl or bl <= 400 then
    gralph.fail("max_firecrawl_body_len<=400 — bodies look like snippets, not clean markdown; " ..
      "the adapter must request scrapeOptions.formats=[markdown] and normalize body to the markdown content")
    return false
  end
  return true
end)
