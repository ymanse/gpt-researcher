# tier-a build harness — what each gate proves

gralph profile: `tier-a.yaml`. Linear 19-node graph:
`s1-red → s1-green → s1-smoke → s2-red → … → s6-smoke → final-verify → DONE`.
A stage advances **only** when its Lua gate mechanically verifies tool-emitted evidence —
never on the agent's self-report. Evidence lives in `no_read/evidence/` (gitignored); loop
state in `.gralph/tier-a/` (gitignored).

## Prereqs

- `gralph` v0.1.0 on PATH (`C:\Users\jskim3\go\bin`)
- host venv: `venv/Scripts/python` with `pytest`, `pytest-asyncio`, `mcp` installed
- docker stack `D:/docker/gptr-mcp/docker-compose.yml` (service `gptr-mcp`, host port 8123);
  the container bind-mounts `D:/dev_ext/gpt-researcher/gpt_researcher` and
  `D:/dev_ext/gptr-mcp/server.py` — **source edits are live only after `--force-recreate`**,
  which the smoke scripts do themselves
- both repos on branch `feature/tier-a-upgrade` (cut from `feat/claude-agent-subscription`)
- `plain python` on PATH (gates shell out via cmd.exe: `python scripts/<x>.py` — relative,
  forward-slash, unquoted; the scripts themselves re-invoke the venv python where needed)

## Run

```sh
cd D:/dev_ext/gpt-researcher/harness
gralph validate tier-a.yaml          # lint
./run-until-done.sh                  # resume-to-DONE outer loop (usage-limit aware)
gralph status --profile tier-a.yaml  # cursor + failure counters
```

`run-until-done.sh` absorbs Claude usage limits **outside** gralph (sleep on the CLI's limit
banner, resume; never a GALP `launcher:` — broken on Windows). `STALL_LIMIT=2` no-progress
rounds → STUCK handoff to a human, honoring the "스테이지당 실패 재시도 초과 시 halt+사유" rule
together with per-node `fail_threshold: 3` session rotation and `--max-iterations` per round.

**Completion alarm**: the loop calls `notify()` at DONE / STUCK / TIMEOUT — terminal bell +
Windows toast (`BurntToast` → `msg *` → bell). The loop is a separate process; gralph alone
only prints `cursor is DONE` to stderr. To be re-invoked automatically, launch it from a
Claude Code session with `run_in_background: true`; for push, swap `notify()`'s body for a
`telegram`/`ntfy` curl.

## Gates

### `sN-red` (`red_common.lua`) — RED integrity (law 3)
Reads `stageN_red.json`, which `pytest_evidence.py` derives from pytest's **own junit xml**:
- `stage` tag matches the node (agent can't submit another stage's file)
- `errors:0` (a test that won't import is not a valid RED), `collected≥1`, `passed:0`, `failed≥1`
- `test_files` sha256 map non-empty — the baseline the GREEN gate hashes against

### `sN-green` (`green_common.lua`) — GREEN anti-tamper, recomputed in-gate
The gate does **not** read agent-submitted numbers: it runs `verify_green.py` itself
(`io.popen`) and parses the script's stdout. The script recomputes from disk:
- `hash_match` — sha256 of every RED-recorded test file equals the RED record
  (tests byte-unchanged; the only honest way to green is implementing)
- `red_collected_ok` — collected count never shrinks below RED (no test deleted)
- stage suite green: `failed=errors=skipped=0`, `collected≥1` (skipping is not passing)
- **full cumulative `tests/tier_a` suite green** — the integration guarantee: any stage
  regressing an earlier stage fails here, every stage, not just at the end
This harness has no refactor loop, so there is no `refactor-certified` baseline: byte-identity
is the only accepted tests baseline. If a legitimate cross-stage test refactor ever becomes
necessary, re-run `pytest_evidence.py --phase red` is NOT the answer (it would demand RED on
green code) — pause the loop and re-baseline by hand-reviewing, then regenerate evidence.

### `sN-smoke` (`smoke_common.lua` + `sN_smoke.lua`) — live container + commit
Reads `stageN_smoke.json`, written only by `smoke_stageN.py` from the container's own
behavior: `recreated:true` (compose `--force-recreate` exit 0), `health:200` (polled), then
per-stage thresholds harvested from **docker logs** (`TIERA_EVIDENCE stage=N k=v` lines the
implementation emits on the live path) and/or host-mounted artifacts:

| stage | live call | gate thresholds |
|---|---|---|
| 1 | `quick_search` (general query) | `firecrawl_sources≥1`, `max_firecrawl_body_len>400` |
| 2 | `deep_research` | `learnings_count≥6` (max over extraction calls; baseline cap was 3) |
| 3 | `deep_research` | `total_claims≥1`, `grounded_ratio_pct≥70`, `unverified` reported |
| 4 | `quick_search` (academic query) | `papers_count≥3`, `papers_in_window≥1`, `from_citers_count≥1` |
| 5 | `deep_research` ×2 (scope on/off) | `scope_on_brief_present`, `brief_len≥1`, `scope_off_ok` (off-call clean = 회귀 0) |
| 6 | `deep_tree_research` (depth 2, 8 nodes) | `tree_json_present`, `report_md_present`, `max_depth_reached≥2`, `node_count≥3`, `pruned_count` reported, `citations_resolve`, `budget_respected` |

Then the gate runs `check_commit.py` **in-gate**: a commit containing `[tier-a][stageN]`
must exist on `feature/tier-a-upgrade` and the tracked tree must be clean (stages 5–6 also
check `D:/dev_ext/gptr-mcp`). COMMIT is therefore folded into the smoke gate: evidence-pass
without a commit still fails.

### `final-verify` (`final_verify.lua`) — capstone, DONE on pass
Runs `final_verify.py` in-gate: full `tests/tier_a` suite green, all 18 evidence files
present + stage-tagged, all 6 stage commits present, trees clean.

## Laws → design mapping (선택 근거)

- **Fail closed / count-bound (law 1)**: every gate pairs its "no failure" checks with a
  positive count (`collected≥1`, `firecrawl_sources≥1`, `node_count≥3`); a missing file or
  field is always FAIL with a prescriptive reason.
- **Tool-emitted evidence (law 2)**: junit xml → `pytest_evidence.py`/`verify_green.py`;
  docker logs + health endpoint + host artifacts → `smoke_*.py`; git → `check_commit.py`.
- **Hollow zero (law 4)**: smoke evidence declares `recreated`/`health` so a threshold can
  never be judged against a stale or dead container; green counts come from a junit file the
  gate's own subprocess produced at HEAD (`staleness impossible` — no index involved).
- **Scanner self-verification (law 5)**: `verify_green.py` treats a missing junit as
  `errors=1` (parse failure ≠ clean), and the suite/stage double-run cross-checks each other
  (`suite_collected ≥ stage_collected`).
- **In-gate recompute**: GREEN/final verdicts are recomputed by the gate's own `io.popen`
  call — a forged evidence file cannot advance the cursor.
- **Law 8**: every gate field appears verbatim in its node's `guidance:` result contract and
  in this file. Change one → change all three in the same commit.
- **Law 10**: production evidence is machine-generated (`pytest_evidence.py`,
  `verify_green.py`, `smoke_stageN.py`, `final_verify.py`) and idempotent — re-running any
  of them is the regenerator that overwrites a hand-edited lie.

## Residual risks (stated, not hidden)

- The loop agent has repo write access, so it *could* edit `harness/scripts/` or forge a
  `TIERA_EVIDENCE` log line. Mitigations: guidance forbids it, `harness/` is tracked on the
  feature branch (any tampering shows in `git diff`/history), and re-running the emitters
  regenerates honest evidence. Review `git log -p harness/` before trusting a finished run.
- `TIERA_EVIDENCE` numbers are emitted by the implementation itself; the pytest suites are
  what pin their semantics (mocked, deterministic), the smoke proves the live path executes.
- Smoke thresholds gate one live sample; flaky external services (firecrawl outages) can
  fail a healthy implementation — rerun the smoke script before debugging.

## Ops notes

- `gralph try <node> --profile tier-a.yaml --report <path>` is a dry-run (no store/cursor
  writes) — use it after hardening any gate, in BOTH directions (accept good, reject bad).
- Gate proof status (2026-07-24, pre-first-run probes): red/green/smoke gates proven in both
  directions (green end-to-end through a real pytest run; smoke PASS via a temporary
  `[tier-a][stage1]` commit, since removed). `final-verify` is proven in the FAIL direction
  only — its PASS direction is first exercised on the live run (single successor, no routing
  risk).
- Probe hygiene: probes overwrite real evidence — regenerate with the emitter scripts, never
  by hand. Pre-first-run probes must end with `no_read/evidence/` and `.gralph/` deleted.
- Cursor rewind: edit `cursor` in `.gralph/tier-a/state.json` to re-run tail stages after a
  gate hardening (there is no set-cursor command).
- Scope expansion (a stage 7+ later): append nodes to the YAML, rewind cursor to the first
  new node. Do NOT re-run completed stages — their tests are green and can't satisfy RED.
