-- d0-resynth gate — the offline re-synthesis runner must be a faithful stand-in for live.
--
-- Everything downstream measures the runner instead of a $330-per-query live run, so if
-- the runner drifts from live the loop optimises a fiction (law 4/5). The decisive field
-- is therefore NOT "resynth.py exists" but bytes_identical: re-synthesising each captured
-- tree must reproduce that run's own report BYTE FOR BYTE. That is reachable because the
-- assembly path is deterministic (create_chat_completion appears twice in
-- tree_research.py, both upstream of the roll-up), so anything less means the sidecar
-- lost state and nobody would see the drift.
--
-- Recomputed IN-GATE by resynth_check.py; a forged evidence file cannot survive it.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local out = L.popen("python scripts/resynth_check.py", "resynth_check.py")
if not out then return end
local function n(k) return L.num(out, k .. "=(%-?%d+)") end

local cq, gc = n("corpus_queries"), n("golden_count")
if not cq or not gc or gc < 5 or cq ~= gc then
  gralph.fail("d0: corpus_queries=" .. tostring(cq) .. " of golden_count=" .. tostring(gc) ..
    " — every golden needs a captured tree + report + resynth sidecar. RUN (FOREGROUND, hours): " ..
    "python scripts/capture.py. Probe the runner on ONE query first " ..
    "(python scripts/capture.py --only outbox-failure-modes, ~330 credits) — a full capture " ..
    "against a runner that is not finished yet wastes ~5,800 credits")
  return
end
if n("corpus_fresh") ~= 1 then
  gralph.fail("d0: the corpus was captured by DIFFERENT implementation bytes than the ones on " ..
    "disk now — its report.md files are the fidelity reference, so they must come from the code " ..
    "being proven. Either revert the change or re-capture: python scripts/capture.py --recapture")
  return
end
if n("shares_assembly") ~= 1 then
  gralph.fail("d0: scripts/resynth.py must CALL the skill's assembly, not copy it — extract the " ..
    "citation-map/roll-up/report-assembly block out of TreeResearchSkill.run() into a method " ..
    "`assemble_report` and have both run() and resynth.py call it. A private copy passes fidelity " ..
    "today and silently diverges the moment s9 edits the real one")
  return
end
local bi = n("bytes_identical")
if bi ~= cq then
  gralph.fail("d0: bytes_identical=" .. tostring(bi) .. "/" .. cq .. " — offline re-synthesis does " ..
    "NOT reproduce the live report. The resynth sidecar is missing state the assembly reads: node " ..
    "INSERTION ORDER (citation ids are assigned in it), per-node sources, learnings, answer_md, " ..
    "answer_digest, status, parent_id/children, and read_docs. See the char offset in " ..
    "no_read/evidence/d0_resynth.json errors[]")
  return
end
if n("netblocked") ~= 0 then
  gralph.fail("d0: the offline re-synthesis attempted " .. tostring(n("netblocked")) ..
    " outbound connections (scripts/netguard counted them from inside the process) — the roll-up " ..
    "must not retrieve. Remove the call, or persist what it was fetching into the sidecar")
  return
end
local rd = n("read_docs_min")
if not rd or rd < 10 then
  gralph.fail("d0: read_docs_min=" .. tostring(rd) .. " < 10 — a corpus whose scraped documents are " ..
    "empty makes a passing fidelity hollow: citation attribution reads read_docs, so with none " ..
    "present both sides agree on nothing")
  return
end
local sc = n("suite_collected")
if not sc or sc < 83 then
  gralph.fail("d0: suite_collected=" .. tostring(sc) .. " < 83 — tests were deleted. d0 refactors " ..
    "shared implementation code, so tests/search_quality + tests/tier_a must both still collect")
  return
end
if n("suite_failed") ~= 0 or n("suite_errors") ~= 0 or n("suite_skipped") ~= 0 then
  gralph.fail("d0: the existing suites are not green (failed=" .. tostring(n("suite_failed")) ..
    " errors=" .. tostring(n("suite_errors")) .. " skipped=" .. tostring(n("suite_skipped")) ..
    ") — the extraction changed behaviour. Fix the implementation, never the tests")
  return
end
if n("frozen_ok") ~= 1 then
  gralph.fail("bench/golden, baseline_firecrawl.json or score_report.py drifted from the s0 freeze " ..
    "— they are READ-ONLY; revert them (git checkout), never re-freeze")
  return
end
if not L.check_commit(0, "d") then return end
