-- d2-live gate — one golden through the real container: does the merge hold end to end,
-- and did the offline measurement tell the truth?
--
-- The agreement check runs FIRST and routes back to d0, because a gap invalidates every
-- other number on this page: it means the fidelity proof d0 established has stopped
-- holding and the loop has been optimising figures live never produces. Fixing the merge
-- in response to a lying instrument is the wrong repair.
--
-- Three successors -> this gate always routes (law 9).
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local blob = L.slurp("no_read/evidence/d2_live.json")
if not blob then
  gralph.fail("no_read/evidence/d2_live.json not found — RUN IN THE FOREGROUND and wait for it " ..
    "(~12 min, ~330 credits): python scripts/live_dedup.py. Backgrounding it orphans the run when " ..
    "the session ends (measured: 8 iterations lost that way)")
  return
end
if not L.must(blob, '"recreated":true',
    "the container must be force-recreated so the bind-mounted source is live — rerun live_dedup.py") then return end
if not L.must(blob, '"health":200',
    "/health did not return 200 after recreate — check docker logs gptr-mcp-server") then return end

local fpout = L.popen("python scripts/code_fp.py", "code_fp.py")
if not fpout then return end
local fp = fpout:match("code_fp=(%x+)")
if not fp then gralph.fail("code_fp.py printed no fingerprint"); return end
if not blob:find('"code_fp":"' .. fp .. '"', 1, true) then
  gralph.fail("d2 evidence was produced by DIFFERENT implementation bytes than the ones on disk now " ..
    "(current code_fp=" .. fp .. ") — re-run: python scripts/live_dedup.py")
  return
end
if not L.check_frozen() then return end
-- [sq][s9], not a d2 tag: the code under test is s9's, and what actually protects the
-- measurement is the CLEAN TRACKED TREE this also requires.
if not L.check_commit(9) then return end

-- how many times this gate has already sent work back, from the append-only journal
local out = L.popen("python scripts/loop_audit.py --instance dedup", "loop_audit.py")
if not out then return end
local spent = L.num(out, "live_journal_d2=(%d+)")
local granted = L.num(out, "grant_live_d2=(%d+)") or 0
if not spent then
  gralph.fail("loop_audit.py did not report live_journal_d2 — run it manually and fix what it reports")
  return
end
local effective = spent - granted
local function send_back(target, why)
  if effective >= 3 then
    gralph.fail("d2: " .. effective .. " live re-routes already spent (counted from journal.jsonl, " ..
      granted .. " human-granted) and it still fails: " .. why .. ". This cap hands the decision to " ..
      "a HUMAN. Do NOT edit .gralph/ state or reset a counter. STOP and let a human read " ..
      "no_read/evidence/d2_live.json")
    return
  end
  gralph.store.set("d2_reason", why)
  gralph.route(target)
end

-- 1. is the instrument honest? (checked before anything it measures)
local rg = L.num(blob, '"ratio_gap":(%d+)')
local lg = L.num(blob, '"lifted_gap":(%d+)')
if not rg or not lg then
  gralph.fail('d2: evidence must carry numeric ratio_gap and lifted_gap — run ' ..
    'scripts/offline_dedup.py first so there is a d1 row to compare against, then live_dedup.py')
  return
end
if rg > 10 or lg > 2 then
  send_back("d0-resynth", "offline<->live DISAGREE (ratio_gap=" .. rg .. " lifted_gap=" .. lg ..
    ") — the offline runner is not reproducing what the live pipeline does, so d0's fidelity proof " ..
    "no longer holds. Fix the runner/sidecar and RE-CAPTURE the corpus; do not touch the merge until " ..
    "the instrument agrees with reality")
  return
end

-- 2. does the merge hold live?
local ln = L.num(blob, '"live_lifted_nodes":(%d+)')
local lr = L.num(blob, '"live_synthesis_ratio_pct":(%d+)')
local lh = L.num(blob, '"live_headings":(%d+)')
local s1 = L.num(blob, '"live_S1_pct":(%-?%d+)')
local s2 = L.num(blob, '"live_S2_pct":(%-?%d+)')
local s3 = L.num(blob, '"live_S3_pct":(%-?%d+)')
if not ln or not lr or not lh or not s1 or not s2 or not s3 then
  gralph.fail('d2: evidence must carry numeric live_lifted_nodes, live_synthesis_ratio_pct, ' ..
    'live_headings, live_S1_pct, live_S2_pct and live_S3_pct — re-run scripts/live_dedup.py')
  return
end
-- live_lifted_nodes is REPORTED, not gated, for the same reason as d1's lifted_nodes_max:
-- the captured corpus shows every kept node's 5-grams are 98-100% unique against every
-- other node, so the threshold demanded deleting or re-wording >=30% of every node's own
-- wording rather than removing duplication. lifted_gap above still uses it, because there
-- it compares OFFLINE against LIVE for the same query -- an instrument check, not a
-- quality bar.
if lr > 70 then
  send_back("s9-impl", "live_synthesis_ratio_pct=" .. lr .. "% > 70% (measured 129% before the fix)")
  return
end
if lh < 4 then
  send_back("s9-impl", "live_headings=" .. lh .. " < 4 (measured 2 before the fix)")
  return
end
-- quality floors: the merge may not buy brevity with facts or with citation grounding
-- 95, not 88: the concatenating report for THIS query scores exactly 100 on the frozen
-- scorer over the captured corpus (measured 2026-07-28), so the live merge gets the same
-- 5-point allowance d1's s2_min_delta gives every query. The old 88 came from the round-4
-- benchmark aggregate and let this query drop 12 points unnoticed.
if s2 < 95 then
  send_back("s9-impl", "live_S2_pct=" .. s2 .. " < 95 — facts were lost in the merge " ..
    "(the concatenating roll-up scores 100 on this query)")
  return
end
if s1 < 95 then
  send_back("s9-impl", "live_S1_pct=" .. s1 .. " < 95 — citation integrity fell (99 before). " ..
    "Every previous rewrite-the-merge design erased [id] grounding this way, one measured all the " ..
    "way to citations_total=0 — see the synthesize_node docstring before trying another one")
  return
end
if s3 > 0 then
  send_back("s9-impl", "live_S3_pct=" .. s3 .. " > 0 — the merge pulled a trap value into the report")
  return
end

gralph.store.set("d2_reason", "")
gralph.route("dedup-audit")
