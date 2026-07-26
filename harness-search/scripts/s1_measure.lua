-- s1-measure gate: retriever recovery, ALL 5 golden queries live in-container.
-- Thresholds: queries_run>=5, scraped_pages_min>=3, retriever_errors_total==0,
-- context_chars_median>=20000 (observed broken median was 1.3-8KB).
dofile(gralph.profile_dir .. "/scripts/measure_common.lua")(1, "s2-red", function(blob, L)
  local qr = L.num(blob, '"queries_run":(%d+)')
  if not qr or qr < 5 then
    gralph.fail("s1 measure: queries_run=" .. tostring(qr) .. " < 5 — every golden query must produce an " ..
      "SQ_PROBE line; fix the errors[] in s1_measure.json and re-run measure.py --stage 1")
    return false
  end
  local sp = L.num(blob, '"scraped_pages_min":(%d+)')
  if not sp or sp < 3 then
    gralph.fail("s1 measure: scraped_pages_min=" .. tostring(sp) .. " < 3 — at least one golden query still " ..
      "scrapes fewer than 3 pages; the retriever/scraper path is still broken for it")
    return false
  end
  if not L.must(blob, '"retriever_errors_total":0',
      "retriever errors are still occurring live (tavily 432 / wikipedia lang-code / scrape failures) — fix the retriever, not the probe") then return false end
  local cc = L.num(blob, '"context_chars_median":(%d+)')
  if not cc or cc < 20000 then
    gralph.fail("s1 measure: context_chars_median=" .. tostring(cc) .. " < 20000 — context starvation persists " ..
      "(observed broken baseline was 1.3-8KB); more/cleaner pages must reach the research context")
    return false
  end
  return true
end)
