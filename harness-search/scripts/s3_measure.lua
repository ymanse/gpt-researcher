-- s3-measure gate: empty-context fail-closed, measured on the measure_pair.
-- Thresholds: queries_run>=2, traps_hit_total==0 (the unit-level proof that an empty
-- context flips a node to FAILED lives in the s3 RED tests, already gated at s3-impl).
dofile(gralph.profile_dir .. "/scripts/measure_common.lua")(3, "s4-red", function(blob, L)
  local qr = L.num(blob, '"queries_run":(%d+)')
  if not qr or qr < 2 then
    gralph.fail("s3 measure: queries_run=" .. tostring(qr) .. " < 2 — both measure_pair queries must complete; " ..
      "fix errors[] then re-run measure.py --stage 3")
    return false
  end
  if not L.must(blob, '"traps_hit_total":0',
      "a golden trap pattern matched the report — an empty-handed node is still answering from prior knowledge; nodes below the context threshold must become FAILED and stay out of synthesis") then return false end
  return true
end)
