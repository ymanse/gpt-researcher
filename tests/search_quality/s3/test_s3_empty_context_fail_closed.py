"""RED tests for s3 — empty-context fail-closed (spec/search-quality.md, defect 3).

Observed: a node whose research scraped nothing still gets an LLM answer written
from the model's prior knowledge and lands in the tree as NodeStatus.ANSWERED;
that fabricated text then feeds the post-order roll-up and the final report,
which is the direct cause of the trap (S3 false-positive) hits.
gpt_researcher/skills/tree_research.py only ever transitions a node to FAILED
when research_node raises — an empty-handed research pass is not an exception.

Contract pinned here (implemented in s3-impl):
  (a) research_node leaves the node FAILED — never ANSWERED — when the node
      researcher read 0 documents, or when the research context is shorter than
      tree_research.MIN_CONTEXT_CHARS. The answer LLM is not called at all, so a
      prior-knowledge answer cannot exist to be salvaged later.
  (b) a FAILED node contributes no text to synthesis/roll-up — neither its answer
      nor the question fallback synthesize_node uses when there is no answer —
      and none of its URLs become citations. Real child summaries below a failed
      node survive: their findings were researched.
  (c) a node with real context and real read documents is unaffected: ANSWERED,
      answer text in the report, read-and-quoted source still cited.

Deterministic, no network: GPTResearcher and create_chat_completion are patched
at the tree_research module seam (its docstring names that seam), embed_question
is replaced on the instance, and CitationAgent only ever sees URLs whose text is
already in the read-document cache, so it never scrapes.
"""
import contextlib
from types import SimpleNamespace

URL_GOOD = "https://good.example.com/doc"
URL_TRAP = "https://trap.example.com/doc"

# the answer sentences appear verbatim in their document, so the s2 read-and-quoted
# narrowing keeps the source and the citation survives — that is the (c) baseline
GOOD_S1 = "Bun ported 530,000 lines of Zig to Rust in eleven days."
GOOD_S2 = "The port used a Claude Code agent harness with git worktree sharding."
GOOD_ANSWER = f"{GOOD_S1} {GOOD_S2}"
GOOD_DOC = ("Engineering blog archive. " + GOOD_S1 + " " + GOOD_S2
            + " Further prose about the sixteen thousand compile error work queue. ") * 6

# what an empty-handed node writes from prior knowledge today; distinctive enough
# that its presence anywhere in a report is unambiguous
FABRICATED = "Zorbulate framistan sprockets emitted 4.2 gigawatts during the port."
TRAP_DOC = ("Unrelated marketing prose. " + FABRICATED
            + " More unrelated marketing prose. ") * 6

ROOT_Q = "root question"
STARVED_Q = "Which vendors ship a marmalade cartography stalactite module?"

ANSWER_BY_Q = {ROOT_Q: GOOD_ANSWER, STARVED_Q: FABRICATED}

# comfortably above any sane per-node context threshold, below research_node's 60k cut
RICH_CONTEXT = "Collected page text from the scraped sources. " * 1200


def _llm_response(answer: str) -> str:
    return f"ANSWER: {answer}\nDIGEST: {answer}\nLEARNINGS:\n- {answer}\n"


def _make_skill(monkeypatch, plan, llm_prompts=None):
    """plan: {question: (research_context, [research_source dicts])}."""
    import gpt_researcher.skills.tree_research as tree_mod

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()
            self._context, self._sources = plan.get(query, ("", []))

        async def conduct_research(self):
            self.visited_urls.update(s["url"] for s in self._sources)
            return self._context

        def get_research_sources(self):
            return [dict(s) for s in self._sources]

        def get_costs(self):
            return 0.0

    async def fake_chat(messages=None, **kwargs):
        content = " ".join(str(m.get("content", "")) for m in (messages or []))
        if llm_prompts is not None:
            llm_prompts.append(content)
        if "Generate up to" in content:  # the expansion prompt, not the answer prompt
            return f"Question: {STARVED_Q}"
        for question, answer in ANSWER_BY_Q.items():
            if question in content:
                return _llm_response(answer)
        return _llm_response("Unmapped question answer.")

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
    return tree_mod.TreeResearchSkill(parent)


async def _research(skill, question):
    """Research one standalone node; return it with whatever terminal status it got.

    Failure may be signalled by status or by raising — both are fail-closed, and
    what these tests pin is the node's terminal status, not the mechanism.
    """
    import gpt_researcher.skills.tree_research as tree_mod

    node = tree_mod.ResearchNode(id="0", question=question, parent_id=None, depth=0)
    skill.nodes[node.id] = node
    with contextlib.suppress(Exception):
        await skill.research_node(node)
    return node


# ---------------------------------------------------------------------------
# (a) a node that researched nothing must FAIL, and must never be asked to write
# ---------------------------------------------------------------------------

async def test_zero_read_documents_fails_closed_without_asking_the_llm(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    prompts = []
    skill = _make_skill(monkeypatch, {STARVED_Q: ("", [])}, prompts)

    node = await _research(skill, STARVED_Q)

    assert node.status is tree_mod.NodeStatus.FAILED, (
        "observed defect 3: a node that scraped 0 documents is left ANSWERED; "
        "an empty-handed research pass must transition to FAILED"
    )
    assert prompts == [], (
        "the answer LLM must not be called at all when research came back empty — "
        "asking it is exactly how prior-knowledge text enters the tree"
    )
    assert not node.answer_md, "a failed node must carry no answer text"
    assert node.sources == []


async def test_long_context_with_zero_read_documents_still_fails_closed(monkeypatch):
    """The threshold must be judged on real research output, not on a proxy that
    can be fat while the research actually failed: a context string can be long
    (boilerplate, error text, an LLM-written preamble) with nothing scraped."""
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch, {STARVED_Q: (RICH_CONTEXT, [])})

    node = await _research(skill, STARVED_Q)

    assert node.status is tree_mod.NodeStatus.FAILED, (
        "0 read documents means the node has no evidence, however many characters "
        "of context text came back"
    )


async def test_context_length_threshold_is_the_answered_failed_boundary(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod
    from gpt_researcher.skills.tree_research import MIN_CONTEXT_CHARS

    assert 1000 <= MIN_CONTEXT_CHARS <= 20000, (
        "the per-node context floor must be a real threshold: below ~1000 chars it "
        "waves through the starved nodes it exists to catch, and above ~20000 it "
        "fails every node, which passes the s3 trap measure hollowly (empty report)"
    )

    docs = [{"url": URL_GOOD, "title": "good", "raw_content": GOOD_DOC}]
    filler = "Collected page text from the scraped sources. " * 1200
    thin = filler[:MIN_CONTEXT_CHARS - 1]
    fat = filler[:MIN_CONTEXT_CHARS + 5000]

    starved = await _research(_make_skill(monkeypatch, {STARVED_Q: (thin, docs)}),
                              STARVED_Q)
    assert starved.status is tree_mod.NodeStatus.FAILED, (
        "context below MIN_CONTEXT_CHARS is context starvation (defect 2) — the "
        "node must fail closed instead of answering from prior knowledge"
    )

    healthy = await _research(_make_skill(monkeypatch, {ROOT_Q: (fat, docs)}), ROOT_Q)
    assert healthy.status is tree_mod.NodeStatus.ANSWERED, (
        "no regression: a node with context at or above the threshold still answers"
    )
    assert healthy.answer_md.strip() == GOOD_ANSWER
    assert healthy.sources == [URL_GOOD]


# ---------------------------------------------------------------------------
# (b) a FAILED node's text never reaches the roll-up
# ---------------------------------------------------------------------------

async def test_failed_node_contributes_no_text_to_synthesis(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch, {})
    node = tree_mod.ResearchNode(id="0", question=STARVED_Q, parent_id=None, depth=0)
    node.status = tree_mod.NodeStatus.FAILED
    node.answer_md = FABRICATED
    node.answer_digest = FABRICATED
    child_summary = "A real child finding that must survive its parent's failure."

    out = await skill.synthesize_node(node, [child_summary], {})

    assert "zorbulate" not in out.lower(), (
        "a FAILED node's answer text must not be rolled up — it is exactly the "
        "prior-knowledge text s3 exists to keep out of the report"
    )
    assert "marmalade cartography" not in out.lower(), (
        "synthesize_node falls back to node.question when there is no answer; a "
        "FAILED node must contribute nothing at all, question included"
    )
    assert child_summary in out, (
        "children that researched successfully must not be dropped with their parent"
    )


async def _run_tree(monkeypatch):
    """Two-node tree: a healthy root expands into one context-starved child."""
    skill = _make_skill(monkeypatch, {
        ROOT_Q: (RICH_CONTEXT,
                 [{"url": URL_GOOD, "title": "good", "raw_content": GOOD_DOC}]),
        # read a document, but came back with no context: starved, not empty-handed
        STARVED_Q: ("", [{"url": URL_TRAP, "title": "trap", "raw_content": TRAP_DOC}]),
    })

    async def fake_embed(text):
        return [1.0, 0.0]

    skill.embed_question = fake_embed
    return await skill.run(query=ROOT_Q, max_depth=1, max_breadth=1, max_nodes=5)


async def test_failed_child_is_excluded_while_the_answered_root_survives(monkeypatch):
    result = await _run_tree(monkeypatch)
    nodes = result["tree"]["nodes"]
    report = result["report_md"]

    # (c) no regression — the node with real context and real documents is untouched
    assert nodes["0"]["status"] in ("answered", "expanded")
    assert GOOD_S1 in report, "the researched root answer must still reach the report"

    # (a) + (b) — the starved child failed and left nothing behind
    assert nodes["0.0"]["status"] == "failed", (
        "the context-starved child must be recorded FAILED in tree.json, not ANSWERED"
    )
    assert "zorbulate" not in report.lower(), (
        "a FAILED node's prior-knowledge answer must not survive into the report"
    )
    assert "marmalade cartography" not in report.lower(), (
        "nor its question, via synthesize_node's no-answer fallback"
    )


async def test_failed_node_sources_never_become_citations(monkeypatch):
    result = await _run_tree(monkeypatch)
    cited = set(result["citation_map"].values())

    assert URL_GOOD in cited, "no regression: the answered node's read source is cited"
    assert URL_TRAP not in cited, (
        "a FAILED node's URLs must not enter the citation map — otherwise its "
        "evidence leaks into the report through the Citations list even when its "
        "text is excluded from the roll-up"
    )
