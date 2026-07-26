"""RED tests for s4 — coverage-oriented expansion + embedding novelty
(spec/search-quality.md, defects 4 and 5).

Observed:
  * `generate_child_questions` hands the expansion LLM nothing but a flat list of
    question STRINGS (`[n.question for n in self.nodes.values()][:30]`) and asks
    for "gaps left by the answer" — a node-local, surface-level derivation. The
    tree's actual covered ground (what each answered node LEARNED) is never
    shown, a FAILED node's question is listed as if the tree had covered it, and
    the arbitrary [:30] dict slice drops covered nodes silently.
  * `compute_novelty` scores a node by string-set overlap of its learnings, so a
    child that restates an existing question in different words looks perfectly
    novel. Every observed tree.json therefore reports `pruned_count: 0`
    (bun-rust-port live run: node_count 13, pruned_count 0) — pruning has never
    once fired in production.

Contract pinned here (implemented in s4-impl):
  (a) the expansion prompt carries the tree's ALREADY-COVERED ground — for every
      node that actually researched, its question AND its findings (digest or
      learnings) — anchored to the root research query, and steers the model
      toward what is NOT yet covered. A FAILED node covered nothing: neither its
      question nor its prior-knowledge findings may be presented as covered, or
      the hole it left is fenced off from ever being re-covered. Coverage is not
      silently truncated below the tree's own max_nodes.
  (b) `compute_novelty` is embedding-cosine based over question embeddings: a
      node semantically near a question already in the tree scores low novelty
      however differently it is worded, and repeated learning STRINGS no longer
      drive the score. A node's own registered embedding must not be scored
      against itself — cosine 1.0 with itself would prune the entire tree.
      The root question counts as covered ground like any other node's.
  (c) end to end through `run()`: a child below `novelty_threshold` becomes
      NodeStatus.PRUNED, is never expanded, contributes no text to the report,
      and is counted in `meta.pruned_count` (> 0 — the observed defect is that
      this is ALWAYS 0). The novel branch is untouched and still expands.

Deterministic, no network: GPTResearcher and create_chat_completion are patched
at the tree_research module seam (its docstring names that seam) and
`embed_question` is replaced on the instance with a fixed question->vector table,
so every cosine in these tests is exact and hand-checkable.
"""
import inspect
from types import SimpleNamespace

# --------------------------------------------------------------------------
# questions. NEAR_Q restates ROOT_Q with almost no shared vocabulary — that is
# the whole point of (b): string matching calls it novel, the embedding does not.
# --------------------------------------------------------------------------
ROOT_Q = "How did the porting team move a large Zig codebase over to Rust?"
NEAR_Q = "What approach let those engineers migrate the sizeable Zig source base into Rust?"
FAR_Q = "Which fuzzing and security review layers validated the ported result?"
GRAND_Q = "What did the adversarial reviewer agent reject most often?"

# fixed embeddings. |NEAR_Q| == 1 and cos(NEAR_Q, ROOT_Q) == 0.8: high enough
# that 1-cos == 0.20 sits below the 0.30 novelty floor, low enough to stay under
# DEDUP_COSINE (0.92) so the child is really CREATED and then PRUNED, rather
# than dropped at the dedup step (which would leave pruned_count at 0 again).
EMBEDDINGS = {
    ROOT_Q: [1.0, 0.0, 0.0, 0.0],
    NEAR_Q: [0.8, 0.6, 0.0, 0.0],
    FAR_Q: [0.0, 0.0, 1.0, 0.0],
    GRAND_Q: [0.0, 0.0, 0.0, 1.0],
}
DEFAULT_EMBEDDING = [0.0, 1.0, 0.0, 0.0]

ROOT_A = "The team split the port across git worktree shards driven by an agent harness."
# distinctive enough that its presence anywhere in a report is unambiguous
NEAR_A = "Kumquat velocipede telemetry saturated the migration ledger."
FAR_A = "Fuzzing, security review and the full test suite gated every merged shard."
GRAND_A = "The adversarial reviewer rejected unchecked lifetime transfers most often."
ANSWER_BY_Q = {ROOT_Q: ROOT_A, NEAR_Q: NEAR_A, FAR_Q: FAR_A, GRAND_Q: GRAND_A}

# only the ANSWER prompt is built from research context — the marker tells the
# two create_chat_completion call sites apart without pinning expansion wording,
# which s4-impl is expected to rewrite.
CONTEXT_MARK = "Collected page text from the scraped sources."
RICH_CONTEXT = (CONTEXT_MARK + " ") * 400  # ~18k chars, well over MIN_CONTEXT_CHARS

NOVELTY_THRESHOLD = inspect.signature(
    __import__("gpt_researcher.skills.tree_research", fromlist=["x"]).TreeResearchSkill.run
).parameters["novelty_threshold"].default


def _llm_response(answer: str) -> str:
    return f"ANSWER: {answer}\nDIGEST: {answer}\nLEARNINGS:\n- {answer}\n"


def _make_skill(monkeypatch, prompts=None):
    """A TreeResearchSkill whose node research, LLM and embeddings are all fixed."""
    import gpt_researcher.skills.tree_research as tree_mod

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()
            self._answer = ANSWER_BY_Q.get(query, "Unmapped question answer.")
            self._url = f"https://example.com/{abs(hash(query)) % 10000}"

        async def conduct_research(self):
            self.visited_urls.add(self._url)
            return RICH_CONTEXT

        def get_research_sources(self):
            return [{"url": self._url, "title": "doc",
                     "raw_content": (self._answer + " Supporting prose. ") * 8}]

        def get_costs(self):
            return 0.0

    async def fake_chat(messages=None, **kwargs):
        content = " ".join(str(m.get("content", "")) for m in (messages or []))
        if prompts is not None:
            prompts.append(content)
        if CONTEXT_MARK in content:  # the ANSWER prompt
            for question, answer in ANSWER_BY_Q.items():
                if question in content:
                    return _llm_response(answer)
            return _llm_response("Unmapped question answer.")
        # the EXPANSION prompt: FAR_Q first — once it exists, ROOT_Q is in the
        # prompt too as covered ground, so ROOT_Q alone no longer identifies it
        if FAR_Q in content:
            return f"Question: {GRAND_Q}"
        if ROOT_Q in content:
            return f"Question: {NEAR_Q}\nQuestion: {FAR_Q}"
        return ""

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    parent = SimpleNamespace(
        query=ROOT_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    skill = tree_mod.TreeResearchSkill(parent)

    async def fake_embed(text):
        return list(EMBEDDINGS.get(text, DEFAULT_EMBEDDING))

    skill.embed_question = fake_embed
    return skill


def _add_node(skill, node_id, question, status, learnings=(), depth=1,
              parent_id: "str | None" = "0"):
    """Register a node the way run() would: in skill.nodes, with its question
    embedding recorded both on the node and in the skill's embedding registry."""
    import gpt_researcher.skills.tree_research as tree_mod

    node = tree_mod.ResearchNode(id=node_id, question=question, parent_id=parent_id,
                                 depth=depth)
    node.status = status
    node.learnings = list(learnings)
    node.answer_digest = " ".join(learnings)
    node.answer_md = node.answer_digest
    node.question_embedding = list(EMBEDDINGS.get(question, DEFAULT_EMBEDDING))
    skill.nodes[node_id] = node
    registry = getattr(skill, "_embeddings", None)
    if registry is not None:
        registry.append(list(node.question_embedding))
    return node


def _mark_learnings_seen(skill, learnings):
    seen = getattr(skill, "_seen_learnings", None)
    if seen is not None:
        seen.update(" ".join(str(l).lower().split()) for l in learnings)


async def _novelty(skill, node):
    """compute_novelty may become a coroutine once it consults embeddings."""
    out = skill.compute_novelty(node)
    return await out if inspect.isawaitable(out) else out


# ---------------------------------------------------------------------------
# (a) the expansion prompt must carry the tree's covered ground
# ---------------------------------------------------------------------------

async def test_expansion_prompt_carries_what_the_tree_already_covered(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    prompts = []
    skill = _make_skill(monkeypatch, prompts)
    root = _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED,
                     ["The port ran on a Claude Code agent harness."],
                     depth=0, parent_id=None)
    sibling = _add_node(skill, "0.0", FAR_Q, tree_mod.NodeStatus.ANSWERED,
                        ["Fuzzing and an adversarial reviewer gated each shard."])
    expanding = _add_node(skill, "0.1", GRAND_Q, tree_mod.NodeStatus.ANSWERED,
                          ["Lifetime transfers were the most rejected change."])

    await skill.generate_child_questions(expanding)

    assert len(prompts) >= 1, "generate_child_questions must call the expansion LLM"
    prompt = prompts[-1]

    assert ROOT_Q in prompt, (
        "the root research query anchors what 'uncovered' is measured against — "
        "without it the expansion drifts into node-local trivia (defect 4)"
    )
    for covered in (root, sibling):
        assert covered.question in prompt, (
            f"a researched node's question is covered ground: {covered.question!r} "
            "must be shown to the expansion LLM"
        )
        assert (covered.learnings[0] in prompt or covered.answer_digest in prompt), (
            "observed defect 4: only question STRINGS are passed, so the model "
            "never learns what the tree actually FOUND and can only derive "
            f"surface variations. {covered.id!r}'s findings must be in the prompt"
        )
    assert "cover" in prompt.lower(), (
        "the prompt must frame the material as already-covered ground and steer "
        "toward what is NOT yet covered — 'fill gaps left by the answer' asks for "
        "a node-local derivation, not tree-level coverage"
    )


async def test_failed_node_is_not_presented_as_covered_ground(monkeypatch):
    """A context-starved node (s3) researched nothing. Listing it as covered
    fences the expansion away from a topic the tree never actually covered, and
    its 'findings' are prior-knowledge text the report is not allowed to chase."""
    import gpt_researcher.skills.tree_research as tree_mod

    prompts = []
    skill = _make_skill(monkeypatch, prompts)
    _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED,
              ["The port ran on a Claude Code agent harness."], depth=0, parent_id=None)
    starved_q = "Which vendors ship a marmalade cartography stalactite module?"
    fabricated = "Zorbulate framistan sprockets emitted 4.2 gigawatts during the port."
    _add_node(skill, "0.0", starved_q, tree_mod.NodeStatus.FAILED, [fabricated])
    expanding = _add_node(skill, "0.1", FAR_Q, tree_mod.NodeStatus.ANSWERED,
                          ["Fuzzing and an adversarial reviewer gated each shard."])

    await skill.generate_child_questions(expanding)
    prompt = prompts[-1]

    assert "marmalade cartography" not in prompt.lower(), (
        "a FAILED node covered nothing — presenting its question as covered "
        "ground stops any child from ever re-covering the hole it left"
    )
    assert "zorbulate" not in prompt.lower(), (
        "a FAILED node's learnings are prior-knowledge text written over missing "
        "evidence (s3); they must never be shown as something the tree found"
    )


async def test_covered_ground_is_not_silently_truncated(monkeypatch):
    """`[:30]` over dict order drops covered nodes with no signal, so the model
    is told to avoid overlapping questions it was never shown."""
    import gpt_researcher.skills.tree_research as tree_mod

    prompts = []
    skill = _make_skill(monkeypatch, prompts)
    _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED,
              ["The port ran on a Claude Code agent harness."], depth=0, parent_id=None)
    questions = [f"Covered sub-question number {i} about the port toolchain?"
                 for i in range(33)]
    for i, question in enumerate(questions):
        _add_node(skill, f"0.{i}", question, tree_mod.NodeStatus.ANSWERED,
                  [f"Covered finding number {i}."])
    expanding = _add_node(skill, "0.33", FAR_Q, tree_mod.NodeStatus.ANSWERED,
                          ["Fuzzing and an adversarial reviewer gated each shard."])

    await skill.generate_child_questions(expanding)
    prompt = prompts[-1]

    missing = [q for q in questions if q not in prompt]
    assert not missing, (
        f"{len(missing)} of {len(questions)} covered questions were dropped from "
        "the expansion prompt. A tree runs to max_nodes (40) nodes, so a 30-item "
        "slice silently hides real coverage from the model that is supposed to "
        f"steer around it. First dropped: {missing[:2]}"
    )


# ---------------------------------------------------------------------------
# (b) compute_novelty is embedding-cosine based
# ---------------------------------------------------------------------------

async def test_semantically_near_question_scores_low_novelty_despite_wording(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch)
    _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED,
              ["The port ran on a Claude Code agent harness."], depth=0, parent_id=None)
    near = _add_node(skill, "0.0", NEAR_Q, tree_mod.NodeStatus.ANSWERED,
                     ["A worktree-sharded agent harness carried out the migration."])

    novelty = await _novelty(skill, near)

    assert novelty < NOVELTY_THRESHOLD, (
        f"cos(NEAR_Q, ROOT_Q) == 0.8: this node restates a question the tree "
        f"already holds, in different words. String-set novelty scores it 1.0 "
        f"(observed defect 5 — pruned_count is 0 in every measured tree); an "
        f"embedding-cosine novelty must put it under {NOVELTY_THRESHOLD}. "
        f"got {novelty}"
    )


async def test_unrelated_question_stays_novel_despite_repeated_learnings(monkeypatch):
    """Two guards in one: repeated learning STRINGS must no longer drive the
    score (that is the matching the spec replaces), and a node must not be
    scored against its OWN registered embedding — cosine 1.0 with itself would
    prune every node in the tree."""
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch)
    shared = ["Fuzzing, security review and the full test suite gated every shard."]
    _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED, shared,
              depth=0, parent_id=None)
    _mark_learnings_seen(skill, shared)
    # orthogonal to every question in the tree, but its learnings all repeat and
    # _add_node has already put its own embedding in the registry
    far = _add_node(skill, "0.0", FAR_Q, tree_mod.NodeStatus.ANSWERED, list(shared))

    novelty = await _novelty(skill, far)

    assert novelty >= 0.7, (
        "this question is orthogonal (cosine 0) to everything already in the "
        "tree, so it is novel ground regardless of how its learnings were "
        "phrased — and its own embedding, registered when the node was created, "
        f"must be excluded from its own comparison. got {novelty}"
    )


async def test_root_question_counts_as_covered_ground(monkeypatch):
    """The root question is the one topic guaranteed to be covered; if it is not
    embedded and registered, the first generation of children can never prune."""
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch)
    root = _add_node(skill, "0", ROOT_Q, tree_mod.NodeStatus.EXPANDED,
                     ["The port ran on a Claude Code agent harness."],
                     depth=0, parent_id=None)
    root.question_embedding = None  # run() never embeds the root today
    near = _add_node(skill, "0.0", NEAR_Q, tree_mod.NodeStatus.ANSWERED,
                     ["A worktree-sharded agent harness carried out the migration."])

    novelty = await _novelty(skill, near)

    assert novelty < NOVELTY_THRESHOLD, (
        "the root question must be embedded and compared against like any other "
        "covered node — otherwise a child that merely rephrases the original "
        f"query looks novel and is expanded again. got {novelty}"
    )


# ---------------------------------------------------------------------------
# (c) end to end: a low-novelty child is PRUNED, never expanded, and counted
# ---------------------------------------------------------------------------

async def _run_tree(monkeypatch):
    """root -> {NEAR_Q (rephrases the root), FAR_Q (new ground)}; FAR_Q -> GRAND_Q."""
    skill = _make_skill(monkeypatch)
    return await skill.run(query=ROOT_Q, max_depth=2, max_breadth=2, max_nodes=10)


def _by_question(result):
    return {n["question"]: n for n in result["tree"]["nodes"].values()}


async def test_low_novelty_child_is_pruned_never_expanded_and_counted(monkeypatch):
    result = await _run_tree(monkeypatch)
    nodes = _by_question(result)

    assert NEAR_Q in nodes, (
        "the near-duplicate child must still be CREATED (cosine 0.8 is under "
        "DEDUP_COSINE) so that pruning is what removes it — a dedup drop leaves "
        "pruned_count at 0, which is exactly the observed defect"
    )
    near = nodes[NEAR_Q]
    assert near["status"] == "pruned", (
        "a child scoring under novelty_threshold must land in tree.json as "
        f"PRUNED. got {near['status']!r}"
    )
    assert near["children"] == [], (
        "a pruned node is never expanded — expanding it spends the node budget "
        "re-covering ground the tree already holds (defect 4+5 compounded)"
    )
    assert result["tree"]["meta"]["pruned_count"] > 0, (
        "observed defect 5: pruned_count is 0 in every measured tree.json "
        "(bun-rust-port live run: node_count 13, pruned_count 0). It must be "
        "positive when a low-novelty child was actually pruned"
    )
    assert "kumquat velocipede" not in result["report_md"].lower(), (
        "a pruned node's answer must not reach the report"
    )

    # no-regression, asserted here rather than in its own test: pruning must not
    # collapse the tree. A run that prunes everything would satisfy pruned_count
    # > 0 hollowly, so the positive count only means something next to a novel
    # branch that survived, expanded, and still rolled up.
    far = nodes[FAR_Q]
    assert far["status"] in ("answered", "expanded"), (
        f"a child on genuinely new ground is neither pruned nor failed. "
        f"got {far['status']!r}"
    )
    assert GRAND_Q in nodes, "the novel child must still expand"
    report = result["report_md"].lower()
    assert "fuzzing, security review" in report
    assert "adversarial reviewer" in report, (
        "the surviving branch's findings, including its grandchild, still roll up"
    )
