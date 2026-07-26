-- s5-measure gate: rollup contradiction / unsupported-claim scan, on the measure_pair.
-- Thresholds: queries_run>=2, contradictions_total==0, unsupported_claims_total==0.
dofile(gralph.profile_dir .. "/scripts/measure_common.lua")(5, "s6-benchmark", function(blob, L)
  local qr = L.num(blob, '"queries_run":(%d+)')
  if not qr or qr < 2 then
    gralph.fail("s5 measure: queries_run=" .. tostring(qr) .. " < 2 — both measure_pair queries must complete; " ..
      "fix errors[] then re-run measure.py --stage 5")
    return false
  end
  if not L.must(blob, '"contradictions_total":0',
      "the final report still contradicts node answers — the rollup consistency pass must reconcile or drop the claim") then return false end
  if not L.must(blob, '"unsupported_claims_total":0',
      "the final report still contains claims backed by NO node answer — unsupported claims must be removed or re-researched") then return false end
  return true
end)
