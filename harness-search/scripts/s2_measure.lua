-- s2-measure gate: citation integrity on the measure_pair (2 live tree runs).
-- Thresholds: queries_run>=2, S1_min_pct>=80, uncited_ids_total==0 (fail-closed: a single
-- [id] without a citations entry is a fail).
dofile(gralph.profile_dir .. "/scripts/measure_common.lua")(2, "s3-red", function(blob, L)
  local qr = L.num(blob, '"queries_run":(%d+)')
  if not qr or qr < 2 then
    gralph.fail("s2 measure: queries_run=" .. tostring(qr) .. " < 2 — both measure_pair queries must complete; " ..
      "fix errors[] then re-run measure.py --stage 2 (completed runs are round-cached)")
    return false
  end
  local s1 = L.num(blob, '"S1_min_pct":(%d+)')
  if not s1 or s1 < 80 then
    gralph.fail("s2 measure: S1_min_pct=" .. tostring(s1) .. " < 80 — cited anchors still don't resolve in " ..
      "re-fetched sources; node.sources must be the documents actually read and quoted, not retriever returns")
    return false
  end
  if not L.must(blob, '"uncited_ids_total":0',
      "the report still contains [id] references with no citations entry — fail-closed rule: zero unbacked citations") then return false end
  return true
end)
