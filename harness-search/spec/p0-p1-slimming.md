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

1. `agent_purpose` is a ContextVar, NOT a thread-local. It must survive both the
   `await` chain (answer/children) and `smart_retriever`'s
   `ThreadPoolExecutor` + `asyncio.run` hop — the latter works because the context
   manager is entered INSIDE the worker thread, around `_run_coro_blocking`.
2. Sibling isolation: `asyncio.gather` copies the context per Task, so one node's
   purpose must never be observed by a concurrent node.
3. Unattributed calls count as `"untagged"`. A site that never fires may be absent
   from the dict or present as 0 — tests must accept both.
4. Retries charge the same site once per attempt. This is intended: the metric is
   sessions spawned, not logical operations.
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

### P1.4 children are generated only for nodes the budget can research
In `TreeResearchSkill.run`, before `generate_child_questions`:

```
nodes_left = max_nodes - researched
calls_left = (agent_budget_limit() - agent_calls_spent()) - synthesis_reserve   # bounded runs only
time_left  = time_budget_s - (time.time() - start)
capacity   = min(applicable limits, expressed in nodes)
headroom   = max(0, capacity - len(frontier))
accepted   = min(max_breadth, headroom)
```

- `headroom == 0` skips the `generate_child_questions` call entirely (that is the saving).
- Per-node cost estimates come from MEASUREMENT once a node has completed
  (calls/node, seconds/node); the seed constants are only used before that.
- An unbounded run (`agent_budget_limit() == 0`) must not have its breadth reduced by
  the calls term.
- Nodes are still PRUNED / created exactly as today for the nodes that ARE expanded —
  s4's prune-before-research contract and s8 are unchanged.

---

## Gates

| gate | value |
|------|-------|
| existing suite | `tests/search_quality/` + `tests/test_agent_budget_reserve.py` all green (baseline 158) |
| calls per researched node | <= 3 |
| `classify` + `choose_agent` per run | <= 2 total |
| pending ratio | <= 10% |
| failed ratio | not worse than baseline (planner bypass shrinks context; watch `MIN_CONTEXT_CHARS`) |
| live cross-check | container `*.jsonl` growth == `agent_calls_this_run` |
