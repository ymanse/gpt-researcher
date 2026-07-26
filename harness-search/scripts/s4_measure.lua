-- s4-measure gate: coverage-driven expansion + embedding novelty, on the measure_pair.
-- Thresholds: queries_run>=2, S4_min_pct>=85, pruned_count_total>=1 (positive evidence
-- pruning actually fires — observed baseline was pruned_count always 0), s5_improved==1
-- (mean S5_pct over the pair strictly above the frozen baseline mean).
dofile(gralph.profile_dir .. "/scripts/measure_common.lua")(4, "s5-red", function(blob, L)
  local qr = L.num(blob, '"queries_run":(%d+)')
  if not qr or qr < 2 then
    gralph.fail("s4 measure: queries_run=" .. tostring(qr) .. " < 2 — both measure_pair queries must complete; " ..
      "fix errors[] then re-run measure.py --stage 4")
    return false
  end
  local s4 = L.num(blob, '"S4_min_pct":(%d+)')
  if not s4 or s4 < 85 then
    gralph.fail("s4 measure: S4_min_pct=" .. tostring(s4) .. " < 85 — required primary domains are still missing " ..
      "from the cited sources; expansion must target uncovered areas")
    return false
  end
  local pc = L.num(blob, '"pruned_count_total":(%d+)')
  if not pc or pc < 1 then
    gralph.fail("s4 measure: pruned_count_total=" .. tostring(pc) .. " — pruning NEVER fired across the pair " ..
      "(the exact observed defect); embedding novelty must actually prune at least one low-novelty child")
    return false
  end
  if not L.must(blob, '"s5_improved":1',
      "mean S5_pct over the measured pair did not exceed the frozen baseline mean — expansion is not improving area coverage") then return false end
  return true
end)
