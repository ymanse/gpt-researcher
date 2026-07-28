-- d1-offline gate — the roll-up must SYNTHESIZE the tree, measured over the frozen corpus.
--
-- Same numbers and same scanner as the s7 gate (scripts/rollup_scan.py), but re-synthesised
-- offline so an iteration costs minutes and ZERO Firecrawl credits instead of 2.6 hours and
-- ~5,800. Measured before any fix, on all 5 goldens: lifted == kept EXACTLY (12/12 on the
-- worst query), max lift 100%, report 120-133% of the kept answers, 2-6 headings. Every
-- kept node answer is pasted in whole; the merge count is zero.
--
-- The S2 floor is the anti-cheat and points the OTHER way from every other gate here: a
-- redundancy metric is cheapest to satisfy by DELETING content, so fact recall from the
-- frozen scorer may not fall below what the concatenating version already achieved.
--
-- Two successors -> this gate always routes (law 9): a metric miss goes back to s9-impl
-- (capped, journal-counted), everything passing goes on to the live confirm.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local blob = L.slurp("no_read/evidence/d1_offline.json")
if not blob then
  gralph.fail("no_read/evidence/d1_offline.json not found — RUN (FOREGROUND): " ..
    "python scripts/offline_dedup.py")
  return
end

-- freshness: the numbers must describe the implementation on disk right now
local fpout = L.popen("python scripts/code_fp.py", "code_fp.py")
if not fpout then return end
local fp = fpout:match("code_fp=(%x+)")
if not fp then gralph.fail("code_fp.py printed no fingerprint"); return end
if not blob:find('"code_fp":"' .. fp .. '"', 1, true) then
  gralph.fail("d1 evidence was produced by DIFFERENT implementation bytes than the ones on disk " ..
    "now (current code_fp=" .. fp .. ") — the roll-up changed after the measurement. Re-run: " ..
    "python scripts/offline_dedup.py")
  return
end

local rf = L.num(blob, '"resynth_failed":(%d+)')
if rf ~= 0 then
  gralph.fail("d1: resynth_failed=" .. tostring(rf) .. " — scripts/resynth.py crashed on at least " ..
    "one golden; see errors[] in no_read/evidence/d1_offline.json")
  return
end
-- "offline" is a claim until the vendor's own balance says so (999 = balance unreadable)
local cd = L.num(blob, '"credits_delta":(%-?%d+)')
if cd ~= 0 then
  gralph.fail("d1: credits_delta=" .. tostring(cd) .. " — the offline loop spent Firecrawl credits " ..
    "(or the balance could not be read, which is not the same as zero). Re-synthesis must replay " ..
    "cached node answers, never retrieve")
  return
end

-- hollow-zero guards: an empty scan must never read as "no redundancy"
local q = L.num(blob, '"queries_scanned":(%d+)')
if not q or q < 5 then
  gralph.fail("d1: queries_scanned=" .. tostring(q) .. " < 5 — every golden must be re-synthesised " ..
    "and scanned; a partial scan cannot prove the roll-up stopped concatenating")
  return
end
local na = L.num(blob, '"node_answers_scanned_total":(%d+)')
if not na or na < 20 then
  gralph.fail("d1: node_answers_scanned_total=" .. tostring(na) .. " < 20 — the corpus carries almost " ..
    "no node answers, so a lift count of 0 would prove nothing (the live baseline scanned 62)")
  return
end

if not L.check_frozen() then return end
if not L.check_commit(9) then return end

-- metric verdicts: a miss is not a gate error, it is a refit round back to s9-impl
local miss, why = nil, ""
local lifted = L.num(blob, '"lifted_nodes_max":(%d+)')
local ratio = L.num(blob, '"synthesis_ratio_pct_max":(%d+)')
local heads = L.num(blob, '"headings_min":(%d+)')
local s2 = L.num(blob, '"s2_aggregate_pct":(%d+)')
local s2d = L.num(blob, '"s2_min_delta":(%-?%d+)')
if not lifted or not ratio or not heads or not s2 or not s2d then
  gralph.fail('d1: evidence must carry numeric lifted_nodes_max, synthesis_ratio_pct_max, ' ..
    'headings_min, s2_aggregate_pct and s2_min_delta — re-run scripts/offline_dedup.py')
  return
end
-- lifted_nodes_max is REPORTED, not gated. Measured 2026-07-28 on the captured corpus:
-- every kept node's 5-grams are 98-100% UNIQUE against every other node, so no other
-- node's text can supply them. Requiring all-but-one node under 70% therefore demands
-- that >=30% of EVERY node's own wording be deleted or re-worded — and keeping a claim's
-- original wording is precisely what keeps its [id] grounded (three rewrite designs lost
-- citations, one to citations_total=0). The threshold was inherited from the s7 gate by
-- analogy, never derived; gating on it forced deletion, which is the one outcome this
-- harness exists to refuse. What it was for -- catching pure concatenation -- is covered
-- by synthesis_ratio_pct_max, since concatenation measures 120-133%.
if s2d < -5 then
  miss, why = "s2_min_delta", "s2_min_delta=" .. s2d .. " < -5 — at least one query LOST facts " ..
    "against its own concatenating baseline (per_query S2_base_pct/S2_delta name it). The " ..
    "aggregate hides this: a merge measured 83 aggregate while dropping 13 and 12 points on the " ..
    "two weakest queries. Deleting content is not de-duplication"
elseif s2 < 80 then
  miss, why = "s2_aggregate_pct", "s2_aggregate_pct=" .. s2 .. " < 80 — facts were LOST while " ..
    "removing redundancy. Deleting content is not de-duplication; every fact the concatenating " ..
    "report carried must survive the merge"
elseif ratio > 70 then
  miss, why = "synthesis_ratio_pct_max", "synthesis_ratio_pct_max=" .. ratio .. "% > 70% — the " ..
    "report is still at least as long as the answers of the nodes the roll-up may use (baseline " ..
    "120-133%). NOTE the denominator is KEPT nodes only. Merge overlapping findings; do NOT reach " ..
    "the number by truncating sections"
elseif heads < 4 then
  miss, why = "headings_min", "headings_min=" .. heads .. " < 4 — the report must be ORGANISED into " ..
    "titled sections so a reader sees each topic once (baseline: an 81k-char report under 2 headings)"
end

if not miss then
  gralph.store.set("d1_miss", "")
  gralph.route("d2-live")
  return
end

-- Round count from gralph's OWN append-only journal, never the store (2026-07-27: an agent
-- blocked by a cap reset the store counter and rewound the cursor).
local out = L.popen("python scripts/loop_audit.py --instance dedup", "loop_audit.py")
if not out then return end
local spent = L.num(out, "off_journal_d1=(%d+)")
local granted = L.num(out, "grant_off_d1=(%d+)") or 0
if not spent then
  gralph.fail("loop_audit.py did not report off_journal_d1 — run it manually and fix what it reports")
  return
end
local effective = spent - granted
if effective >= 3 then
  gralph.fail("d1: " .. effective .. " offline refit rounds already spent (counted from " ..
    "journal.jsonl, " .. granted .. " human-granted) and the roll-up still misses " .. miss ..
    ". This cap hands the decision to a HUMAN. Do NOT edit .gralph/ state, do not rewind the " ..
    "cursor, do not reset a counter — the journal records it anyway. STOP and let a human read " ..
    "no_read/evidence/d1_offline.json. (" .. why .. ")")
  return
end
gralph.store.set("d1_miss", miss)
gralph.route("s9-impl")
