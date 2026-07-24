-- smoke_common.lua — shared SMOKE+COMMIT gate skeleton.
-- Usage from sN_smoke.lua:  dofile(...)(N, function(blob, L) ...stage checks... end)
-- Verifies: evidence exists for THIS stage, container was force-recreated and /health
-- returned 200 (both recorded by the smoke script, which is the only writer of the file),
-- then the per-stage threshold checks, then IN-GATE commit verification via check_commit.py
-- (git log for [tier-a][stageN] on feature/tier-a-upgrade + clean tracked tree; stages 5-6
-- also check the gptr-mcp repo).
return function(stage, checks)
  local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
  local rel = "no_read/evidence/stage" .. stage .. "_smoke.json"
  local blob = L.slurp(rel)
  if not blob then
    gralph.fail(rel .. " not found — RUN: python scripts/smoke_stage" .. stage .. ".py"); return
  end
  if not L.must(blob, '"stage":' .. stage .. ',',
      "evidence is for the wrong stage — rerun smoke_stage" .. stage .. ".py") then return end
  if not L.must(blob, '"recreated":true',
      "the container must be recreated (docker compose --force-recreate) so the bind-mounted source is live — rerun the smoke script") then return end
  if not L.must(blob, '"health":200',
      "/health did not return 200 after recreate — the container is not healthy; check docker logs gptr-mcp-server") then return end

  if not checks(blob, L) then return end

  local p = io.popen("python scripts/check_commit.py --stage " .. stage)
  if not p then gralph.fail("could not launch check_commit.py"); return end
  local out = p:read("*a"); p:close()
  if not out or not out:find("commits_ok=1", 1, true) then
    local why = (out or ""):match("reason=([^\n]+)") or "check_commit.py produced no verdict"
    gralph.fail("[tier-a][stage" .. stage .. "] commit gate: " .. why ..
      " — commit ALL tracked changes on feature/tier-a-upgrade with that message prefix, then resubmit")
    return
  end
end
