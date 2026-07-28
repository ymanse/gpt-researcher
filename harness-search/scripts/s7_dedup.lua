-- s7-dedup gate — the roll-up must SYNTHESIZE the tree, not concatenate it.
--
-- Measured before any fix (round 4, all 5 goldens): EVERY kept node answer is carried into
-- the report >=70% verbatim (lifted == kept exactly, 5/5 queries, max lift 100%), and the
-- report runs 120-133% of the kept answers -- all of them plus a preamble, no merging.
-- Upstream cause (see no_read/audit/pending_rca.md): expansion is blind to the PENDING
-- queue, so siblings research near-identical questions and their answers overlap by
-- construction. That is the redundancy a reader sees;
-- a sentence-level dedup scan reports ~0% on the same files because each pasted answer
-- is internally unique prose and the overlap between siblings is topical, not lexical.
--
-- The S2 check is the anti-cheat: the cheapest way to satisfy any redundancy metric is
-- to delete content, so fact recall (frozen scorer) may not fall below what the
-- concatenating version already achieved.
local L = dofile(gralph.profile_dir .. "/scripts/lib.lua")
local blob = L.slurp("no_read/evidence/s7_dedup.json")
if not blob then
  gralph.fail("no_read/evidence/s7_dedup.json not found — RUN: python scripts/rollup_scan.py --dir <report dir>")
  return
end

-- freshness: the numbers must describe the implementation on disk right now
local fpout = L.popen("python scripts/code_fp.py", "code_fp.py")
if not fpout then return end
local fp = fpout:match("code_fp=(%x+)")
if not fp then gralph.fail("code_fp.py printed no fingerprint"); return end
if not blob:find('"code_fp":"' .. fp .. '"', 1, true) then
  gralph.fail("s7 evidence was produced by DIFFERENT implementation bytes than the ones on disk now " ..
    "(current code_fp=" .. fp .. ") — re-run python scripts/rollup_scan.py after regenerating the reports")
  return
end

-- hollow-zero guards: an empty scan must never read as "no redundancy"
local q = L.num(blob, '"queries_scanned":(%d+)')
if not q or q < 5 then
  gralph.fail("s7: queries_scanned=" .. tostring(q) .. " < 5 — every golden must be scanned; " ..
    "a partial scan cannot prove the roll-up stopped concatenating")
  return
end
local na = L.num(blob, '"node_answers_scanned_total":(%d+)')
if not na or na < 20 then
  gralph.fail("s7: node_answers_scanned_total=" .. tostring(na) .. " < 20 — the trees carry almost no " ..
    "node answers, so a lift count of 0 would prove nothing (baseline scanned 62)")
  return
end

local lifted = L.num(blob, '"lifted_nodes_max":(%d+)')
if not lifted then gralph.fail('s7: report a numeric "lifted_nodes_max"'); return end
if lifted > 1 then
  gralph.fail("s7: lifted_nodes_max=" .. lifted .. " > 1 — that many node answers are still carried into " ..
    "the report >=70% verbatim (baseline 10). The roll-up must MERGE overlapping node findings into one " ..
    "statement per claim, not append each node's essay under its own heading")
  return
end
local ratio = L.num(blob, '"synthesis_ratio_pct_max":(%d+)')
if not ratio then gralph.fail('s7: report a numeric "synthesis_ratio_pct_max"'); return end
if ratio > 70 then
  gralph.fail("s7: synthesis_ratio_pct_max=" .. ratio .. "% > 70% — the report is still at least as long as " ..
    "the answers of the nodes the roll-up may use (baseline 120-133%: every kept answer plus a preamble, " ..
    "zero merging). NOTE the denominator is KEPT nodes only — pruned/pending answers are excluded from the " ..
    "roll-up by design, and counting them rewarded a tree for pruning more. Merge overlapping findings; " ..
    "do NOT reach the number by truncating sections")
  return
end
local heads = L.num(blob, '"headings_min":(%d+)')
if not heads or heads < 4 then
  gralph.fail("s7: headings_min=" .. tostring(heads) .. " < 4 — the report must be ORGANISED into titled " ..
    "sections so a reader can see each topic once (baseline had a 69k-char report under 3 headings)")
  return
end

-- anti-cheat: shorter must not mean emptier
local s2 = L.num(blob, '"s2_aggregate_pct":(%d+)')
if not s2 then gralph.fail('s7: report a numeric "s2_aggregate_pct" from the FROZEN scorer'); return end
if s2 < 80 then
  gralph.fail("s7: s2_aggregate_pct=" .. s2 .. " < 80 — facts were LOST while removing redundancy. " ..
    "Deleting content is not de-duplication; every fact the concatenating report carried must survive " ..
    "the merge (baseline aggregate was 80)")
  return
end
