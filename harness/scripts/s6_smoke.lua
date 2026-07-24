-- s6_smoke.lua — Stage 6 live smoke: deep_tree_research end-to-end on the live container.
dofile(gralph.profile_dir .. "/scripts/smoke_common.lua")(6, function(blob, L)
  if not L.must(blob, '"tree_json_present":true',
      "tree.json not found under D:/dev_ext/gptr-mcp/outputs — the tool must persist the tree per the artifact contract") then return false end
  if not L.must(blob, '"report_md_present":true',
      "final report .md missing/empty — post-order synthesis must always produce a report") then return false end
  local d = L.num(blob, '"max_depth_reached":(%d+)')
  if not d or d < 2 then
    gralph.fail("max_depth_reached<2 — the tree never expanded past the root's children; check frontier expansion")
    return false
  end
  local nc = L.num(blob, '"node_count":(%d+)')
  if not nc or nc < 3 then
    gralph.fail("node_count<3 — the tree is degenerate; Self-Ask expansion must create child nodes")
    return false
  end
  if not L.must(blob, '"pruned_count":',
      "pruned_count must be reported (0 is fine) — hiding it hides whether novelty pruning ran") then return false end
  if not L.must(blob, '"citations_resolve":true',
      "citation map does not resolve — every report citation [id] must map to a URL in tree.json citations") then return false end
  if not L.must(blob, '"budget_respected":true',
      "budget_respected is not true — node/credit budgets were exceeded; enforce budgets in the frontier loop") then return false end
  return true
end)
