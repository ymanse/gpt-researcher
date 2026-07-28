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

**Blocker, recorded 2026-07-28.** Point 3 cannot be implemented under the current s9 RED
contract. The hash-locked fixture builds nodes with answers and `_read_docs` only, sets
`cfg.strategic_llm_provider` to the literal string `"mock"`, and patches **no** seam —
neither `create_chat_completion` nor an embedding — while its docstring states the
assembly "makes no LLM call". Any model or embedding call inside `assemble_report`
therefore breaks a frozen test, and `verify_impl.py --stage 9` will not accept that.
Note also that `rollup_scan`'s lift is a **set** intersection of 5-grams, so re-ordering
content under themes does not lower `lifted_nodes_max` — only removing or re-wording it
does. With semantics forbidden and deletion caught by `s2_aggregate_pct ≥ 80`, the two
gates form a vise. Re-cutting the RED contract (patched seams, outcomes pinned instead of
mechanism — the `tests/search_quality/s8` pattern) is a **human** decision and is recorded
in `no_read/audit/grants.json` when taken.

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
| `lifted_nodes_max` | ≤ 1 | 12 |
| `synthesis_ratio_pct_max` | ≤ 70 | 120–133% (denominator = **kept** nodes only) |
| `headings_min` | ≥ 4 | 2–6 |
| `s2_aggregate_pct` | ≥ 80 | 80 — the anti-cheat |
| `code_fp` | recomputed in-gate | evidence must describe the bytes on disk now |

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
| `live_lifted_nodes` | ≤ 1 (was 12) | → `s9-impl` |
| `live_synthesis_ratio_pct` | ≤ 70 (was 129) | → `s9-impl` |
| `live_headings` | ≥ 4 (was 2) | → `s9-impl` |
| `live_S2_pct` | ≥ 88 (was 100) | → `s9-impl` — facts lost |
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

*Trap found while reverting — do not use `git checkout` to restore a file here.*
`code_fp` hashes raw bytes, and `git checkout` writes CRLF while the working copy the
corpus was captured against was LF. Restoring an otherwise-identical file that way moved
the fingerprint (`9a587656fc9c992d` → `9fcbf516c2f2e5f3`) and would have read as "the
implementation changed", demanding a ~5,800-credit re-capture for a file whose content had
not changed at all. Restore with `git cat-file blob <sha>:<path>` written verbatim, then
confirm `python scripts/code_fp.py` matches `no_read/dedup/corpus/corpus.json`. The
fingerprint is deliberately left byte-exact (normalising it would invalidate the recorded
manifest for no present gain), so this is a documented handling rule, not a defect to fix.

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
