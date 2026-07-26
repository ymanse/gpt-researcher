# PRD — search-quality build

**Index.** The authoritative spec is [`spec/search-quality.md`](spec/search-quality.md):
observed defect layers (1-6), the S1~S6 metric definitions, the golden-set schema, the
scorer CLI contract, per-stage completion conditions, and the hard rules. The harness
mechanics (what each gate proves, loop policy, probe log) are in [`HARNESS.md`](HARNESS.md).

**Decomposition guidance for s0.** s0 builds the measuring instrument before anything is
built: >=5 golden sets (bun-rust-port mandatory, seeded from
`D:/dev_ext/gptr-mcp/outputs/how-did-bun-port-*`), a deterministic LLM-free scorer, a
good/bad fixture pair that discriminates on all six metrics, and a one-shot firecrawl
multi-agent baseline scored by the same scorer, then frozen. If s0 is weak, every later
score is fiction — which is why its gate is the strictest in the flow.

**Stage order** s1(retriever) → s2(citation) → s3(fail-closed) → s4(coverage/novelty) →
s5(rollup consistency) mirrors the causal chain of the observed defects: context
starvation upstream causes the hallucination/citation defects downstream, so fixing s1
first makes every later measurement meaningful.
