# tier-a upgrade — spec index

Authoritative spec for the autonomous build loop (`tier-a.yaml`):

| Doc | Covers |
|---|---|
| [spec/tier-a-prompt.md](spec/tier-a-prompt.md) | The 6 stage definitions, common gate rules, thresholds — the source of truth for every stage's RED cases and SMOKE assertions |
| [spec/tree-research-tool-design-2026.md](spec/tree-research-tool-design-2026.md) | Full design for Stage 6 `deep_tree_research` (tree persistence, frontier, dedup, synthesis, budgets) |

Stages (already decomposed — the loop is linear, no decompose stage):

1. **firecrawl `/search` in SmartRetriever routing** — new `retrievers/firecrawl/` adapter + ROUTING_TABLE entries
2. **learnings 손실 압축 완화** — `num_learnings` 3→8, `max_tokens` 1000→2500, env-configurable
3. **citation-verification 패스** — CitationAgent: firecrawl re-scrape + quote matching, unmatched → `unverified`
4. **논문 레인 Firecrawl research 승격** — academic routing → paper search + citers expansion + recency window
5. **scope/clarification 옵션화** — `scope: bool` MCP param; on → 1-round brief; off → 현행 유지
6. **`deep_tree_research` 트리 도구** — `TreeResearchSkill` + MCP tool (see design doc)

Fixed decisions (interview, 2026-07-24): branch `feature/tier-a-upgrade` off
`feat/claude-agent-subscription` in BOTH `gpt-researcher` and `gptr-mcp`; pytest on the host
venv; regression baseline = cumulative `tests/tier_a/` suite only; harness lives in
`harness/`; smoke queries + thresholds are hardcoded in `scripts/hconf.py` and the smoke gates.

How each stage is verified: see [HARNESS.md](HARNESS.md).
