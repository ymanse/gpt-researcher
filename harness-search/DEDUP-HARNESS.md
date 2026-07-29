# DEDUP-HARNESS.md — what each dedup.yaml gate proves

Companion to `HARNESS.md` (the search-quality build). Same repo, same frozen instrument,
different question: **the final report must synthesize the tree, not concatenate it.**

Run it with `./run-dedup.sh` (or `run-dedup-detached.cmd`). `gralph status --profile
dedup.yaml` shows the cursor. Loop state lives in `.gralph/dedup/`.

## The defect this harness attacks

Measured on benchmark round 4 (all 5 goldens) and re-confirmed on the 2026-07-28
verification run:

| | round 4 (5 queries) | verify1 (outbox, current code) |
|---|---|---|
| lifted / kept node answers | 7/7, 5/5, 5/5, 10/10, 8/8 — **all equal** | **12/12** |
| max lift | 100% on every query | 100% |
| synthesis ratio (report / kept answers) | 120–133% | **129%** |
| headings | 3–6 | **2** |
| report size | 33k–69k chars | **81k chars** |
| S2 (fact recall, frozen scorer) | aggregate 80 | 100 |

Every kept node answer is carried into the report ≥70% verbatim. The merge count is zero.

**The trap.** A sentence/paragraph near-duplicate scan reports ~0% on these same files
(repeated 5-grams 0–2%, max section-pair Jaccard 0.22). Each pasted answer is internally
unique prose and sibling overlap is *topical*, not lexical. Chasing word similarity
produces a confident "no redundancy" verdict on a report a reader plainly sees repeating
itself. What is measurable, and is the actual defect, is **the paste** — `lifted_nodes`.

**Upstream is already handled and did not fix this.** `no_read/audit/pending_rca.md`
found that expansion was blind to the PENDING queue, so siblings researched near-identical
questions. The mitigation shipped (`_queued_ground`, prune-before-research) and redundancy
did **not** improve — the verify1 report got longer. So this harness fixes the roll-up.

## Why an offline loop, and what makes it trustworthy

A live benchmark round costs ~5,800 Firecrawl credits and 2.6 hours. The roll-up itself
buys nothing: `create_chat_completion` appears exactly twice in
`gpt_researcher/skills/tree_research.py` and both call sites are upstream of it, so once
the node answers and the scraped documents are on disk the whole assembly can be replayed
for free.

That only helps if the replay is *faithful*. If offline and live disagree, the loop spends
its whole budget optimising numbers nobody will ever see (law 4/5). Hence `d0`'s decisive
field is not "does the runner exist" but **`bytes_identical`**: re-synthesising a captured
tree must reproduce that run's own `report.md` byte for byte.

## Node by node

### d0-resynth — the instrument

Deliverables: an additive `<stem>.resynth.json` sidecar from `_persist` (tree.json's
frozen contract is untouched), a shared `assemble_report` method extracted out of `run()`,
and `scripts/resynth.py`. Creating that one runner file is the only exception to the
no-hand-edits-under-`scripts/` rule.

Gate (`scripts/resynth_check.py`, re-run in-gate):

| field | threshold | why |
|---|---|---|
| `corpus_queries` | == golden count (5) | a partial corpus cannot measure the set |
| `corpus_fresh` | 1 | the reference reports must come from the code being proven |
| `shares_assembly` | 1 | the runner CALLS `assemble_report`; a private copy passes today and diverges the moment s9 edits the real one (law 5) |
| `bytes_identical` | == `corpus_queries` | the decisive proof; anything less means the sidecar lost state (node insertion order, sources, learnings, read_docs) |
| `netblocked` | 0 | outbound connections counted from inside the probed process by `scripts/netguard/sitecustomize.py` — the roll-up may not retrieve |
| `read_docs_min` | ≥ 10 | positive binding: an empty `read_docs` makes a passing fidelity hollow (law 1) |
| `suite_collected` | ≥ 83, failed/errors/skipped 0 | d0 refactors shared implementation code; `tests/search_quality` + `tests/tier_a` stay green |
| `frozen_ok` | 1 | golden/baseline/scorer unchanged since the s0 freeze |

Cost discipline, written into the guidance: probe on **one** golden
(`capture.py --only outbox-failure-modes`, ~330 credits) until `bytes_identical=1`, then
capture the rest. Any later edit to `gpt_researcher/` moves `code_fp` and invalidates the
capture, so a full capture against an unfinished runner wastes ~5,800 credits.

### s9-red / s9-impl / s9-review — the merge

Reuses the search-quality lane gates verbatim (`red_common`, `impl_common`,
`review_common` with stage 9), so the RED/GREEN anti-tamper guarantees are the proven
ones: RED must collect and fail (`errors:0, collected≥1, passed:0, failed≥1`, test hashes
recorded), GREEN is recomputed in-gate by `verify_impl.py --stage 9` (hash re-check, stage
+ full cumulative suite, ruff on changed files), and the reviewer is a separate lane that
sees only `review_diff.py --stage 9` output plus the completion conditions.

**What s9 must achieve.** Overlapping node findings become **one statement per claim**,
organised into titled sections. Not: reflowed text, arbitrary heading splits, or a shorter
report bought by dropping content.

Review routing uses `blocking_count`; the implementer clears findings with
`addressed_findings` in `s9_impl_ack.json`, bound to that round by `review_head_sha` so a
stale ack cannot satisfy a new finding by id collision. Three blocking rounds
(`rev_journal_s9`) hand the decision to a human.

**Two constraints that kill naive fixes**, both measured:

1. *Deleting content satisfies every redundancy metric.* `s2_aggregate_pct ≥ 80` from the
   frozen scorer is the only thing standing against it. This gate points the opposite way
   from every other gate in the build — it asks for **less** — so the anti-cheat is not
   optional and must never be relaxed.
2. *Rewriting text erases citation grounding.* The `synthesize_node` docstring records
   three progressively wider "LLM rewrites the merge, then re-derive `[id]` markers by
   fuzzy matching" designs, each of which lost more citations, culminating in
   `citations_total=0` on a 33-node tree. A rewrite must own re-attribution against
   `self._read_docs`; `d2` checks `live_S1_pct ≥ 95`.

### How the literature says to do this

Researched 2026-07-28 after the first s9 cut merged 35 of ~1085 sentences (one golden:
0 of 241) and the review lane showed that what it *did* merge was disproportionately
distinct content. Three findings, in the order they matter:

1. **Derive the outline; do not inherit the tree's.** STORM (arXiv:2402.14207, NAACL
   2024) generates an outline from the collected references and expands it section by
   section. Egnyte's production deep-research agent does the same in its writer stage:
   a thematic meta-analysis over *all* question analyses produces "emergent, overarching
   themes" that become the report's sections, and it dispatches one writing task per
   theme. Today a node's answer can only appear beneath its own node, so two siblings
   restating one finding are never in the same place — no threshold can merge them.
   Themes cut across nodes, which makes the merge structural rather than a similarity
   guess.
2. **Merge by selection, not by rewriting.** Keep one cluster member's original wording
   and migrate the other members' `[id]`s onto it. Nothing is re-worded, so nothing loses
   grounding — the failure mode that killed all three prior designs. The evidence on
   citation timing points the same way: arXiv:2509.21557v2 (four datasets, human eval
   κ=0.873) finds post-hoc attribution beats generation-time on coverage with competitive
   correctness (human-rated answer correctness 78% vs 69%, citation hallucination 37% vs
   41%) and recommends "P-Cite-first … reserving G-Cite for precision-critical settings".
   Contradicting evidence exists (arXiv:2410.11217: post-hoc only helps models that lack
   attribution ability), which is why `live_S1_pct ≥ 95` may not be relaxed either way.
3. **Cluster atomic claims, semantically.** Claim/nugget decomposition is the standard
   unit (Claimify, FActScore, DnDScore; NuggetIndex: "indexing atomic facts … reducing
   redundancy"). The measured redundancy here is topical, not lexical, so any word-overlap
   bar high enough to be safe is too high to fire. That is a ceiling, not a tuning problem.
4. **Similarity selects candidates; equivalence decides.** This is the operator four
   implementations got wrong. Cosine, LSA, word overlap and coverage all score the
   *intersection* of two texts; the merge decision lives in the *difference*. arXiv:
   2509.08304 ("Modeling Semantic Coverage Relations via Answerability") defines the
   relation between two texts as the set relation between their **Answerable Question
   Sets**: equivalence when the sets match, inclusion when one contains the other,
   **overlap** when each has questions the other cannot answer — and overlap must never
   be collapsed. Operationally: *merge A and B iff neither answers a question the other
   cannot.* Two stages keep it affordable — embedding cosine generates candidates, and an
   LLM **enumerates** what each side says that the other does not (enumeration is steadier
   than classification, and the two lists are the explanation the review lane needs).
   Fail closed: an unclear verdict does not merge. Not merging costs
   `synthesis_ratio_pct_max`; wrongly merging costs facts, and `s2_min_delta ≥ −5` is the
   tighter bound. The paper's own benchmark tops out at 61.4% accuracy for general
   relation classification, so this is not a solved problem — but the decision needed here
   is only the empty-difference case on candidate pairs, with the safe default available.

**The seams are open (since the 2026-07-28 re-cut).** The s9 RED fixture replaces
`skill.embed_question` with an offline stand-in and patches `tr.create_chat_completion`,
so the implementation may call a real embedding or model in production while the test
stays deterministic. The previous contract forbade both, which is why it was re-cut.

**Two measured ceilings — do not build a third lexical rule.** Cut 1 (word overlap +
significant figures) merged 35 of ~1085 sentences, one golden 0 of 241. Cut 2 (LSA token
cosine at 0.75) merged 0–4 units per golden out of 139–267, keeping 98–100% of characters
— a no-op that only *added* heading lines and therefore **raised** the ratio. Note also
that `rollup_scan`'s lift is a **set** intersection of 5-grams, so re-ordering content
under themes does not lower `lifted_nodes_max` on its own; only removing or re-wording it
does. Headings alone move one metric of four.

**A no-op merge passes `s9-impl`.** `verify_impl.py --stage 9` checks tests, ruff and the
frozen bench — nothing there distinguishes a merge that worked from one that found
nothing, so the failure surfaces four stages later at `d1-offline`. The impl guidance
therefore requires running `scripts/offline_dedup.py` as a preflight (minutes, zero
credits) and reporting units merged, characters removed and per-node lift over the
captured corpus before submitting.

### d1-offline — measure for free

`scripts/offline_dedup.py` re-synthesises all five goldens with the code on disk and scans
them with `scripts/rollup_scan.py` — the same scanner the s7 gate used, so these numbers
are directly comparable with the table above.

| field | threshold | baseline |
|---|---|---|
| `resynth_failed` | 0 | — |
| `credits_delta` | 0 | read from Firecrawl's own balance before/after; -1 (unreadable) fails closed |
| `queries_scanned` | 5 | hollow-zero guard |
| `node_answers_scanned_total` | ≥ 20 | live baseline scanned 62 |
| `s2_min_delta` | ≥ −5 | per-query, vs that query's own concatenating baseline |
| `s2_aggregate_pct` | ≥ 80 | corpus baseline 88 |
| `synthesis_ratio_pct_max` | ≤ 70 | 120–133% (denominator = **kept** nodes only) |
| `headings_min` | ≥ 4 | 2–6 |
| `lifted_nodes_max` | *reported, not gated* | 12 |
| `code_fp` | recomputed in-gate | evidence must describe the bytes on disk now |

**Why `lifted_nodes_max` stopped being a gate (2026-07-28, human decision).** Measured over
the captured corpus, every kept node's 5-grams are **98–100% unique** against every other
node (`bun-rust-port` 9 nodes: 100×8, 99; `denorm` 10×100; `edge-ai` 7×100; `outbox` 13
nodes: 100×10, 99, 99, 98; `solid-state` 6×100). Siblings share almost no wording — the
redundancy is entirely topical. So no other node's text can supply a node's 5-grams, and
`lifted ≤ 1` demanded that ≥30% of **every** node's own wording be deleted or re-worded.
Re-wording is what destroyed citations three times (`citations_total=0`); deleting is what
`s2_min_delta` refuses. The threshold was inherited from the s7 gate by analogy and never
derived, and gating on it drove three cuts toward deletion — the third dropped 60–68% of
claim units, positionally (keep-rate by decile 83,43,26,29,26,43,23,20,20,11), and shipped
an adjacency swap whose own comment admitted it existed to defeat an ordered 5-gram
measure. What the metric was for — catching pure concatenation — is covered by
`synthesis_ratio_pct_max`, since concatenation measures 120–133%. The number is still
computed and recorded on every run.

**Why fact retention is now per query.** `s2_aggregate_pct` is a mean over five queries, so
one report can lose half its facts and still pass. Measured: a merge scored 83 aggregate
while dropping **13 and 12 points** on `denorm-derived-table` and `edge-ai-face-access` —
the two queries whose concatenating baselines were already the weakest. Each query is now
scored against its own baseline from the frozen corpus (`S2_base_pct` / `S2_delta` in
`per_query`; baselines 100 / 63 / 75 / 100 / 100, aggregate 88), and no query may fall more
than 5 points.

The denominator is kept nodes on purpose: measured against *all* nodes the ratio rewards a
tree for pruning more (one query scored "best" at 47% purely because it pruned 8).

A metric miss is not a gate error — it routes back to `s9-impl` with the reason, capped at
three rounds (`off_journal_d1`, counted from `journal.jsonl`).

### d2-live — does it hold, and did offline tell the truth

One golden (`outbox-failure-modes`, the worst case) through the real container.
**The agreement check runs first**, because a gap invalidates everything else on the page:

| field | threshold | on failure |
|---|---|---|
| `ratio_gap` | ≤ 10 | → `d0-resynth` (the instrument is wrong, not the merge) |
| `lifted_gap` | ≤ 2 | → `d0-resynth` |
| `live_lifted_nodes` | *reported, not gated* | used only by `lifted_gap` above |
| `live_synthesis_ratio_pct` | ≤ 70 (was 129) | → `s9-impl` |
| `live_headings` | ≥ 4 (was 2) | → `s9-impl` |
| `live_S2_pct` | ≥ 95 (concat scores 100) | → `s9-impl` — facts lost |
| `live_S1_pct` | ≥ 95 (was 99) | → `s9-impl` — citation integrity lost |
| `live_S3_pct` | ≤ 0 (was 0) | → `s9-impl` — a trap value entered the report |

Plus `recreated`/`health` 200, an in-gate `code_fp` match, `frozen_ok`, and a `[sq][s9]`
commit with both tracked trees clean — the s9 tag rather than a d2 one, because the code
under test is s9's and what actually protects the measurement is the clean-tree
requirement. Three re-routes (`live_journal_d2`) hand the decision to a human.

### dedup-audit — audit the gates

`scripts/audit_check.py --profile dedup.yaml` verifies `try_ok` (law 6 pass+fail probe for
every node, node list parsed from the profile so it cannot drift), `git_ok` (no commit
after the `[sq][s0]` freeze touches the goldens, the baseline or the scorer), `frozen_ok`,
`regen_ok` (law 10 idempotence), `sync_ok` (law 8: every threshold token appears in the
profile guidance **and** in this file) and `store_untampered`.

The law-8 token set for this profile: `corpus_queries`, `corpus_fresh`,
`bytes_identical`, `netblocked`, `shares_assembly`, `read_docs_min`, `suite_collected`,
`credits_delta`, `resynth_failed`, `queries_scanned`, `node_answers_scanned_total`,
`lifted_nodes_max`, `synthesis_ratio_pct_max`, `headings_min`, `s2_aggregate_pct`,
`live_lifted_nodes`, `live_synthesis_ratio_pct`, `live_headings`, `live_S1_pct`,
`live_S2_pct`, `live_S3_pct`, `ratio_gap`, `lifted_gap`, `code_fp`, `blocking_count`,
`addressed_findings`, `review_head_sha`, `frozen_ok`, `store_untampered`.

## Intervention log

**2026-07-28 — the s9 RED contract was re-cut, by a human.** Rewinding the cursor is the
one act every node's guidance forbids, so it is recorded here and in
`no_read/audit/grants.json` (`rev:s9` = 3) rather than done quietly.

*What happened.* Three s9 implementation rounds merged 35 of ~1085 sentences (one golden:
0 of 241) and the review lane demonstrated, by replaying the rule over this harness's own
corpus, that what they *did* merge was disproportionately distinct content — two product
variants collapsed into one, a third source's disagreement deleted, a bold heading kept
while the finding under it was dropped. That is a ceiling, not a threshold to tune.

*Why the contract had to go.* The frozen fixture built nodes with answers and
`_read_docs`, set `cfg.strategic_llm_provider` to the literal string `"mock"`, and patched
**no** seam, while its docstring asserted the assembly "makes no LLM call". Any model or
embedding call inside `assemble_report` therefore broke a hash-locked test. Since the
redundancy is topical and `rollup_scan`'s lift is a set intersection (re-ordering cannot
lower it), that left only deletion — which `s2_aggregate_pct ≥ 80` correctly refuses. The
referee made the right answer unreachable, which is exactly the law-6 failure, one level
up from a gate bug.

*What was reverted.* `9132f2b0` (RED), `517a44d1`, `da524005`, `0269421a` (impl). Kept:
`65f0db34` (d0 — `assemble_report`, the resynth sidecar) and the captured corpus. The
three spent review rounds are forgiven by the grant because they were spent against the
discarded contract; without that the cap would fire on the first review of the new one.

**2026-07-28 (evening) — three more review rounds granted (`rev:s9` 3 → 6).** The cap fired
correctly and the agent handed off without touching `.gralph/`. Granted because the build is
close and both remaining defects are named with their causes: `synthesis_ratio_pct_max` 68
(gate ≤70, was 129), `headings_min` 7, `lifted_nodes_max` 1 (was 12) and
`s2_aggregate_pct` 83 all pass; the single blocker is `s2_min_delta = −13`
(`denorm-derived-table` 63→50, `edge-ai-face-access` 75→63 against their own baselines).
The second defect is review R1, verified in the diff: the contested path is exempt from
dedup on **both** routes, so contested passages are never compared with each other and
`solid-state-battery` states the same $10B vs $300B+ disagreement three times. Protecting
contested content from *deletion* is not a reason to exempt it from *merging*.

**2026-07-29 — third RED re-cut, by a human.** Four implementations (word overlap, LSA
cosine, coverage/vocabulary-exhaustion, then item-budget selection) each failed at the
same point: they merged on how much two sentences have in common and were blind to what
only one of them said. The last one reached `synthesis_ratio_pct_max` 67 with four of
five queries lossless — real progress — but `denorm-derived-table` lost 2 of its 8 golden
facts (S2 63 → 38, `s2_min_delta` −25), traced to exactly two collapses: a *complete
refresh* sentence merged away in favour of an *incremental refresh* one, and a
closure-table *definition* merged away in favour of a closure-table *trade-off*. Same
topic, different claim, both times.

The RED contract could not express that, so it was re-cut with the real failing pairs as
its fixture, read verbatim from the frozen corpus. `red_common` requires `passed:0`, so
the re-cut necessarily reverted the implementation to the post-d0 state — the 67% ratio is
lost as code and survives only in the commits and reviews. Kept: `65f0db34` (d0), today's
arxiv scraper fix and the tier_a widening of `verify_impl`, none of which are s9 work.

*Trap found while reverting — do not use `git checkout` to restore a file here.*
`code_fp` hashes raw bytes, and `git checkout` writes CRLF while the working copy the
corpus was captured against was LF. Restoring an otherwise-identical file that way moved
the fingerprint (`9a587656fc9c992d` → `9fcbf516c2f2e5f3`) and would have read as "the
implementation changed", demanding a ~5,800-credit re-capture for a file whose content had
not changed at all. Restore with `git cat-file blob <sha>:<path>` written verbatim, then
confirm `python scripts/code_fp.py` matches `no_read/dedup/corpus/corpus.json`. The
fingerprint is deliberately left byte-exact (normalising it would invalidate the recorded
manifest for no present gain), so this is a documented handling rule, not a defect to fix.

### Why the third cut still merged nothing — measured, 2026-07-29

The third RED cut prescribed the right operator (equivalence decides, similarity only
selects) and the implementation still came back at `synthesis_ratio_pct_max` 121 with
`lifted_nodes_max` 10 of 10 — a report that scores exactly its own concatenating
baseline. Splitting the claim unit from a paragraph down to a sentence did not move it.
Three causes, all measured on `denorm-derived-table` (212 claim units, 430 candidate
pairs, 13 screening groups of 16):

1. **Coverage was required to be MUTUAL.** A pair only reached a verdict when both ends
   screened "nothing of my own", and 4 of 32 screened units did — so essentially no pair
   was ever judged. But coverage is directional and the common shape in a research tree
   is a later node restating an earlier finding *with more detail*: A is covered by B,
   B is not covered by A. Deleting A there loses nothing — that is precisely what the
   judge enumerated — and B survives whole. Requiring the mutual case buys no safety and
   refuses most of the real redundancy.
2. **The judge does not quote verbatim.** It strips the markdown from the fragment it
   quotes (`**Fast Refresh using materialized view logs**` comes back bare), so
   `fragment in unit` missed and the whole line — including a `UNIQUE: none` — was
   discarded as unreadable. Measured parse rate: 10 of 16 statements in one screen, 5 of
   8 in another.
3. **The judge drops the keyword.** One screen answered
   `"The refresh method can be incremental or a complete refresh" => none`, with no
   `UNIQUE:` at all. The keyword-anchored line pattern threw it away.

Both parse failures fail closed, which is why they were invisible: a discarded verdict
looks exactly like "the model said this one is unique".

*The counters are the reason this took a round.* `_merge_stats` and `_merge_calls` were
computed, logged, and written to no file, so `screens`, `verdicts` and `screened_out`
never reached evidence and a merge that found nothing was indistinguishable from one
that worked until d1 scored it. They now travel out through `assemble_report`'s result
and `run()`'s `stats.merge`. Wiring a GATE to them needs a change under
`harness-search/scripts/`, which the impl lane is not allowed to make.

### The fourth cut merges, and is still nowhere near the ratio — measured, 2026-07-29

First round in which the merge removed content without losing a fact. Probe at
`code_fp d099fa7f53576633` (`scripts/offline_dedup.py --only denorm-derived-table`):

| | third cut | this cut | d1 gate |
|---|---|---|---|
| `synthesis_ratio_pct` | 119–121 | **116** | ≤ 70 |
| `max_lift_pct` | 99–100 | **91** | *reported* |
| `headings` | 6 | **10** | ≥ 4 |
| `S2_pct` / `S2_delta` | 63 / 0 | **63 / 0** | delta ≥ −5 |
| claim units merged away | 0 of 212 | **22 of 202** | — |
| characters removed | 0 | **2,728 (4.3%)** | — |

Three causes of the previous no-op were mechanical and are fixed (see the commits): the
screening groups were slices of a nearest-neighbour chain through *one* connected
component covering all 212 units, so the judge was asked whether a LISTEN/NOTIFY sentence
says anything a `hierarchyid` sentence does not; the verdict pairs came from a global
top-3 neighbour graph that need not contain the pair the screen had just implicated; and
a cumulative failure counter read three scattered timeouts as a dead judge and abandoned
the screens still queued. Groups are now grown clusters (`MERGE_CLUSTER_FLOOR` 0.30, the
corpus's own top ~1% of pair scores), verdict pairs come from the screen's own group, and
a two-unit screen is used as the verdict it already is.

**What remains is not mechanical, and it is the judge's own threshold.** Across the
screened groups the model answers `UNIQUE: none` for roughly one statement in twenty:
asked what a reader would lose if a statement were deleted, it can nearly always name
*something* — a figure, a qualifier, an example — even when two statements make one
claim. That rate caps the merge at ~10% of units however good the clustering gets, and
the ratio needs ~42% of the report's characters gone. Loosening the prompt (telling it to
disregard extra detail, examples or quantities) is the obvious next lever and is also
exactly how golden facts 2 and 8 were lost before, so it is not a change to make without
a per-query `S2_delta` measurement behind it.

**Also unmeasured here: the offline judge is not the deployment's.** `Config()` resolves
`STRATEGIC_LLM` to the `claude_agent:sonnet` default when the replay runs with
`cwd = harness-search/`, because only `main.py` loads the repo `.env` (which names
`openrouter:minimax/minimax-m3`). A screening call costs 142–320s as a CLI round trip,
which is what makes a five-golden preflight an hour. And `EMBEDDING` resolves correctly
but the OpenAI key returns 429 `insufficient_quota`, so candidate selection degrades to
the bag-of-content-words fallback on every offline run — the semantic half of "similarity
selects candidates" has never actually been exercised.

### The fifth cut: half the units were not claims, and half are never screened — 2026-07-29

Review round 4 filed three mechanical defects against the *unit*, not the judge, and the
last round's ack had attributed the whole residual to the judge's threshold. All three are
fixed and the first two are measurable without a single model call, over the captured
corpus (`no_read/scratch/s9r5_units.py`, `s9r5_groups.py`):

| | before | after |
|---|---|---|
| claim units, 5 goldens | 984 | **863** |
| units with an unbalanced quote / paren / code span | 46 | **0** |
| units opening on a bare anaphor | 90 | **1** |
| `denorm-derived-table` units | 202 | **166** |

Re-synthesised (`scripts/offline_dedup.py --only denorm-derived-table`, `code_fp
53b22255540c1f11`, zero credits):

| | fourth cut | fifth cut | d1 gate |
|---|---|---|---|
| `synthesis_ratio_pct` | 116 | **115** | ≤ 70 |
| `report_chars` | 60,681 | **60,119** | — |
| `max_lift_pct` | 91 | **98** | *reported* |
| `headings` | 10 | **10** | ≥ 4 |
| `S2_pct` / `S2_delta` | 63 / 0 | **63 / 0** | delta ≥ −5 |
| claim units merged away | 22 of 202 | **10 of 166** | — |
| characters removed | 2,728 (4.3%) | **2,979 (5.7%)** | — |

Fewer units absorbed, more text removed: the units are whole claims now, so each merge is
worth more. `max_lift_pct` rose because the fourth cut's 22 absorptions were concentrated
in one node, and some of them were not merges the judge authorised — R2 and R3 were both
manufacturing them. **This is the answer R4 asked for: re-measured on whole claims, the
judge's `UNIQUE: none` rate does not go up.** Equivalence-only merging at this granularity
tops out near 6% of the characters, against a ratio that needs ~42%. The next lever is the
instruction itself (disregard extra detail, examples, quantities), and that is exactly how
golden facts 2 and 8 were lost, so it may not ship without a per-query `S2_delta` behind
it — which is a d1 round, not an impl round.

- **R1 — a unit was cut mid-quotation.** `_SENT_BREAK_RE` breaks after any `[.!?]`
  followed by a non-lowercase character, with no regard for an enclosing quote, paren or
  backtick span, so Oracle's one quoted sentence became two units filed under two
  different headings — one opening a quotation it never closes, the other closing one it
  never opened and starting on a "Thus" with no antecedent. The break is now skipped when
  it would leave a delimiter open, but *only* when a later break in the same paragraph
  closes it again, so one stray delimiter cannot swallow the rest of the block.
- **R6 — and a unit opening on a bare anaphor** ("This is not a disagreement between
  sources…") has its subject in the sentence before it. Same dependency as the colon
  lead-in and now the same fix. The one survivor in the corpus is the first sentence of a
  node answer, which has no predecessor to travel with.
- **R2 — an empty verdict tail read as coverage.** `_reads_as_covered` counted `""` among
  the "nothing of its own" answers, so `"<fragment>" => UNIQUE:` with the content wrapped
  onto the next line — the continuation carries no `=>` and is skipped — deleted the very
  statement the judge had just named content for. Fail-open in the one place the module
  swears it fails closed. An empty tail is now no verdict at all.
- **R3 — the label fallback read the first word of a quoted fragment.** `^\W*([A-Za-z]|
  \d{1,2})\b` ran against a `head` that is normally the quotation itself, so any statement
  opening with a one-letter word (`"A separate table stores…"`, `"a complete refresh…"`,
  `"I found that…"`) was attributed to statement A or I of the screen. A label must now
  sit at the start after nothing but list punctuation and be followed by a label
  delimiter — a quote character disqualifies it.

**The second cap is the clustering, and it is not the judge's fault.** Only units inside
one screening group are ever compared, and on the fixed units the groups reach less than
half the corpus: `bun-rust-port` 87 of 170 (51%), `denorm-derived-table` 64 of 166 (39%),
`edge-ai-face-access` 59 of 132 (45%), `outbox-failure-modes` 124 of 251 (49%),
`solid-state-battery` 84 of 144 (58%). Everything else is a singleton that cannot merge
whatever the judge answers, so even a judge that collapsed every group to one statement
would leave `denorm-derived-table` around 30% shorter against a ratio that needs ~42%.
This is measured with the **bag-of-content-words fallback**, because the OpenAI key still
answers 429 — the semantic candidate selection the design calls for has never run here,
and `MERGE_CLUSTER_FLOOR` 0.30 is a lexical threshold standing in for it. Lowering it is
not the fix: at 0.08 the graph was one connected component and the groups were arbitrary
slices of a chain through every topic (see the third cut).

Left undone, again: **R5, the section titles are still comma-joined keyword bags**
("Moved, entire, contents, counts"). Prose titles are a model call whose output nothing
in the fixture can verify — the s9 stand-in answers any prompt with UNIQUE lines, not an
outline. R1 did remove the half of R5 that made it worse: a section no longer opens on an
orphan quote-tail under a bag heading.

## Loop policy

Stop hierarchy: **gate-pass** > **journal-counted refit rounds** (`rev_journal_s9`,
`off_journal_d1`, `live_journal_d2`, each capped at 3) > **`fail_threshold`** session
rotation > **`--max-iterations`** (`./run-dedup.sh`).

Caps are derived from `.gralph/dedup/journal.jsonl`, which is append-only and
framework-owned, **not** from `store.json`. On 2026-07-27 an agent blocked by the s2
review cap reset the store counter and rewound the cursor; the cap exists to hand a
decision to a human, so it must not be derived from something the blocked party can edit.
Extra rounds are granted only by a human writing them into
`no_read/audit/grants.json` (`rev:s9`, `off:d1`, `live:d2`), which the audit surfaces.

`fail_threshold`: 3 for d0/red/impl, 2 for review/offline/live/audit. Live nodes must run
in the **foreground** — a `claude -p` session ends when it stops emitting, orphaning a
backgrounded run (measured: 8 iterations lost that way on the search-quality build).

**Agent timeout is 120m, raised from 75m on 2026-07-29.** Once the merge started asking a
model for a per-pair verdict, two `s9-impl` sessions were killed at exactly `1h15m0s`
mid-work, losing everything they had not committed. The dominant cost is the preflight:
a tree offers 100–300 candidate pairs and re-synthesising all five goldens multiplies that
by five. `scripts/offline_dedup.py --only <golden>` exists for that — it probes one query
into `d1_offline.probe.json`, deliberately a **separate file**, because the d1 gate reads
`d1_offline.json` and requires `queries_scanned == 5`, so a probe can never be mistaken
for gate evidence. The impl guidance points at `denorm-derived-table`: the other four
goldens are already lossless, so it is the only query that still decides anything.

**Completion alarm.** `run-dedup.sh` calls `notify()` at its `DONE`/`STUCK`/`TIMEOUT`
exits (terminal bell + Windows toast, `BurntToast` → `msg *` → bell). The loop is a
separate process and gralph only prints `cursor is DONE` to stderr, so nothing else
surfaces the finish. Run it under Claude Code with `run_in_background:true` to be
re-invoked on exit, or swap `notify()`'s body for a `telegram`/`ntfy` curl.

## Shared-script changes this harness required

Widened, never re-pointed — stages 1–5 and `search-quality.yaml` behave exactly as before:

- `verify_impl.py`, `review_diff.py`, `pytest_evidence.py`, `regen_evidence.py`:
  `--stage` accepts 1–9 (9 = the dedup lane). `pytest_evidence.py` was the second
  unsatisfiable-gate find of the pre-run pass: `s9-red` tells the agent to run it, and it
  rejected stage 9
- `check_commit.py`: `--stage` 0–9 and a `--prefix s|d` so `[sq][d0]`/`[sq][d2]` are checkable
- `lib.lua`: `L.check_commit(stage, prefix)`, prefix defaults to `s`
- `review_common.lua`: optional `next_node` and `instance` arguments, defaults unchanged.
  Also a real bug fix the law-6 probe matrix surfaced: the stage check demanded
  `"stage":N,` **with** a trailing comma, but `stage` is the last key of that schema, so
  any reviewer emitting sorted-key JSON wrote `"stage":N}` and could never satisfy the
  gate — the broken-referee case law 6 exists to catch. Both terminators are accepted now
  (`measure_common.lua` already carried the same fix)
- `loop_audit.py`: `--instance`, stages 1–9, plus `off_journal_d1` / `live_journal_d2`
- `hconf.store_get(key, default, instance)`
- `try_probe.py`, `audit_check.py`: `--profile`; the audit's node list is now parsed from
  the profile rather than hardcoded, so a new node cannot silently escape the try matrix
