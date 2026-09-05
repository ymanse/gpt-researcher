# P0/P1 — call-site attribution and worker slimming (pinned interface)

Frozen contract for the P0 (measurement) and P1 (slimming) stages of the LangGraph
plan (`D:\dev_ext\langgraph-research-plan-2026.md`). Tests are written against THESE
names. Implementation may not rename them; a rename is a spec change.

Branch: `feature/langgraph-research` (both repos), base `feature/search-quality`.
Run tests as the harness does: `venv/Scripts/python -m pytest <target> -c harness-search/pytest.ini`
(the fork's own pyproject sets `asyncio_mode = strict`, which fails bare `async def` tests).

---

## Why (measured 2026-09-05)

One tree node costs 8-9 `claude` CLI sessions, of which 5-6 buy nothing:
`choose_agent` 1, smart classification 1 inside `plan_research`, `plan_research_outline` 1,
smart classification 3-4 more (one per sub-query), answer 1, children 1.
20 nodes x 8 = 160 against a 100 cap, so the tree is cut short by its own overhead.

Separately, the last 5 tree runs left 67-80% of created nodes PENDING and pruned
exactly 0 — children are generated (an LLM call + an embedding each) for nodes the
budget can never research.

---

## P0 — attribution

`gpt_researcher/llm_provider/claude_agent/_subscription.py`

```python
SITES = ("choose_agent", "classify", "plan", "answer", "children",
         "merge", "verify", "untagged")

@contextlib.contextmanager
def agent_purpose(site: str): ...      # sets a ContextVar; note_agent_call() reads it

def agent_calls_this_run() -> int: ...          # NEW: delta since begin_agent_run()
def agent_calls_by_site() -> dict[str, int]: ... # NEW: delta since begin_agent_run(), per site
```

Rules:

1. `agent_purpose` is a ContextVar, NOT a thread-local. It must survive the `await`
   chain (answer/children) AND `smart_retriever`'s thread hop.

   **Corrected 2026-09-05 (verified empirically, the first draft of this rule was wrong).**
   `_run_coro_blocking` has two branches and the important one breaks a plain ContextVar:

   - Branch A — no loop running on this thread: `asyncio.run(coro)`. Context is intact
     (and `asyncio.to_thread`, used at researcher.py:857, copies context into the worker).
   - Branch B — a loop IS running on this thread (the MCP server path, and the direct
     `retriever_instance.search(...)` calls at researcher.py:656 and :1001):
     `pool.submit(lambda: asyncio.run(coro))`. **`ThreadPoolExecutor.submit` does not copy
     contextvars**, so the new thread starts with an EMPTY context and a purpose set by the
     caller is silently lost. Measured: the var reads `None` there.

   Consequence if unfixed: every production `classify` lands under `"untagged"`, and the
   `classify + choose_agent <= 2 per run` gate passes **vacuously by reading 0**.

   Fix at the submit site, not at the call site — capture the context and run inside it:
   `ctx = contextvars.copy_context()` then `pool.submit(lambda: ctx.run(asyncio.run, coro))`.
   This is general (any future context state rides along) and keeps the purpose where it
   belongs: around the logical operation, on the calling thread.

   The test for this rule must drive the REAL topology (purpose entered on a thread that
   already has a running loop, coroutine handed to `_run_coro_blocking`). A test that
   enters the purpose inside its own worker thread certifies nothing — a plain ContextVar
   passes it while production mis-attributes.
2. Sibling isolation: `asyncio.gather` copies the context per Task, so one node's
   purpose must never be observed by a concurrent node.
3. Unattributed calls count as `"untagged"`. A site that never fires may be absent
   from the dict or present as 0 — tests must accept both.
4. Retries charge the same site once per attempt. This is intended: the metric is
   sessions spawned, not logical operations. (This is a property of the retry wrapper in
   `chat_model`, so it is pinned near the retry path, not in the counter tests.)

4b. **Every tagged site needs an end-to-end test that the real code path attributes.**
   Charging `note_agent_call()` directly only proves the counter works. An implementation
   that ships `SITES`, `agent_purpose` and both delta functions with ZERO call sites
   wrapped goes fully green on counter-only tests and produces
   `agent_calls_by_site() == {"untagged": N}` in production. For each site in the table
   below there must be a test that drives the real function (with the LLM boundary faked)
   and asserts the call landed on that site.
4c. Before any `begin_agent_run()` in a process (the harness imports `gpt_researcher`
   directly and never arms), the baseline is process start: `agent_calls_this_run()` equals
   `agent_calls_spent()` and `agent_calls_by_site()` covers every call so far.

4d. The delta is per-ARMING, not per-run. Two concurrent runs re-baseline each other,
   exactly as `agent_calls_spent()` pools rather than isolates. The live cross-check gate
   (container `*.jsonl` growth == `agent_calls_this_run`) is therefore **only valid for a
   serialized run** — see appendix A.2 of the plan, where 7 concurrent armings shared one
   ceiling of 171 and each run researched a single node.

5. **`agent_calls_spent()` and `agent_budget_limit()` keep their current cumulative,
   pooling semantics** — `tests/test_agent_budget_reserve.py` is a frozen contract on
   them. The delta lives only in the two NEW functions, whose baselines are snapshotted
   by `begin_agent_run()`.

`stats` (tree) and the linear tools gain `agent_calls_this_run` and
`agent_calls_by_site` alongside the existing `agent_calls_spent` / `agent_calls`.

Tagged call sites:
| site | file | what |
|------|------|------|
| `choose_agent` | `actions/agent_creator.py` | agent+role selection |
| `classify` | `retrievers/smart/smart_retriever.py` `_classify_query` | routing category |
| `plan` | `actions/query_processing.py` `plan_research_outline` | sub-query planning |
| `answer` | `skills/tree_research.py` `research_node` | ANSWER/DIGEST/LEARNINGS |
| `children` | `skills/tree_research.py` `generate_child_questions` | follow-ups |
| `merge` | `skills/tree_research.py` claim-equivalence judge | roll-up dedup |
| `verify` | `gptr-mcp/verification.py` | faithfulness audit |

Also: `compute_novelty` logs the cosine it compared against at INFO, so the
prune threshold can be set from a distribution instead of a guess (5 runs, 0 prunes).

---

## P1 — slimming

### P1.1 classify once per run
`TreeResearchSkill` resolves the category ONCE and hands it to every node researcher:

```python
researcher.cfg.smart_retriever_force_category = self._category   # after construction
```

The run-level resolution goes through `_classify_query` **once**, on the run's own query.
Resolving the category some other way (a constant, a keyword rule) is NOT equivalent:
routing quality is the whole point of the category, and one call per run is inside every
tier budget. An implementation that skips the classifier entirely fails this stage.

`SmartRetriever._classify_query` already honours
`getattr(self.cfg, "smart_retriever_force_category", None)` when the value is a key of
`ROUTING_TABLE`; `Config._set_attributes` lowercases `SMART_RETRIEVER_FORCE_CATEGORY`.
An unset / unknown category must fall through to the LLM classifier unchanged.

### P1.2 choose the agent once per run
`TreeResearchSkill` resolves `(agent, role)` once and passes it:
`GPTResearcher(..., agent=self._agent, role=self._role)`.
`ResearchConductor.conduct_research` already guards on
`if not (self.researcher.agent and self.researcher.role)`, so BOTH must be non-empty
or the guard does not fire.

### P1.3 preset sub-queries bypass the planner
New named attribute on `GPTResearcher` (NOT via `**kwargs` — kwargs are forwarded into
`plan_research_outline` and leak into LLM calls):

```python
GPTResearcher(..., preset_sub_queries: list[str] | None = None)
```

`ResearchConductor._get_context_by_web_search` uses it INSTEAD of
`await self.plan_research(...)` when it is a non-empty list, skipping both the planner
LLM call and the planner's own probe search.

Interaction that must be preserved: the existing code appends the original query when
`report_type != "subtopic_report"`. With a preset the researcher's query IS the node
question, so the preset path must not research the same string twice.

`preset_sub_queries=None` (default) must leave today's behaviour byte-identical.

**Read it with `getattr(self.researcher, "preset_sub_queries", None)`, never as a bare
attribute access.** `harness-search`'s own rig (`tests/search_quality/s10/merge_bench.py`)
drives `_get_context_by_web_search` over a `SimpleNamespace` researcher that has no such
attribute; a bare read kills s10 with `AttributeError`, and the s11 fakes cannot catch it
because they always set the attribute explicitly.

### P1.4 do not pay for an expansion the budget can never research

**Corrected 2026-09-05.** The first draft of this rule also clamped the ACCEPTED CHILD
COUNT to the remaining capacity. That is wrong twice over, and it was measured:

- It buys almost nothing. `generate_child_questions` is ONE LLM call that returns all
  candidates at once, so accepting 1 instead of 4 saves three embeddings, not a session.
  The whole saving is in **not making the call**.
- It destroys a deliberate feature. Nodes accepted but never researched stay PENDING and
  are published by `assemble_report` under `## Unresearched Questions` — the s5 work whose
  commit is "the report says what it never researched". Clamping the count deletes those
  questions instead of disclosing them.

Applying the clamp temporarily took the suite from 158 green to 156: it breaks
`s4/test_s4_coverage_and_novelty.py::test_primary_source_question_is_researched_before_a_generic_sibling`
(with `max_nodes=2, max_breadth=2` only the first-emitted candidate would be created, so
the primary-source node never exists) and
`test_disclosure_unresearched.py::test_run_stats_carry_nodes_researched_and_nodes_total`
(with `max_nodes=2, max_breadth=3` only 1 of 3 children is created, so `nodes_total` is 2
instead of 4 and the disclosure section vanishes).

So the rule is only the gate, in `TreeResearchSkill.run` before `generate_child_questions`:

```
nodes_left = max_nodes - researched
calls_left = (agent_budget_limit() - agent_calls_spent()) - synthesis_reserve   # bounded runs only
time_left  = time_budget_s - (time.time() - start)
capacity   = min(applicable limits, expressed in whole nodes)
if capacity - len(frontier) <= 0:
    skip generate_child_questions entirely      # <- the entire saving
else:
    generate, and accept up to max_breadth      # <- UNCHANGED from today
```

- Accepted children are capped by `max_breadth` and nothing else. Today's behaviour.
- Per-node cost estimates come from MEASUREMENT once a node has completed (calls/node,
  seconds/node); seed constants are used only before the first completion. A run whose
  fixtures never charge a call measures 0 calls/node — the calls term must treat an
  unmeasurable cost as "not limiting", never as "capacity 0".
- An unbounded run (`agent_budget_limit() == 0`) is not throttled by the calls term.
- Nodes that ARE expanded still prune and get created exactly as today; s4's
  prune-before-research contract, s4's priority contract, s8, and the disclosure test are
  all unchanged.

---

## Gates

| gate | value |
|------|-------|
| existing suite | `tests/search_quality/` + `tests/test_agent_budget_reserve.py` all green (baseline 158) |
| calls per researched node | <= 3 |
| `classify` + `choose_agent` per run | <= 2 total |
| pending ratio | <= 10% on a run whose budget was not exhausted. NOT zero — PENDING nodes are the disclosed open questions, not waste |
| failed ratio | not worse than baseline (planner bypass shrinks context; watch `MIN_CONTEXT_CHARS`) |
| live cross-check | container `*.jsonl` growth == `agent_calls_this_run`, **serialized run only** |
| test hygiene | any test that calls `begin_agent_run(n)` must `begin_agent_run(0)` on teardown. s11 sorts before s2-s10, so a leaked allowance silently converts every later tree run from unbounded to bounded |
