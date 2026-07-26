"""RED tests for s2 — citation integrity (spec/search-quality.md, observed defect 6a).

Observed: node.sources holds "every URL the retriever returned" instead of
"documents actually read AND quoted" (bun run: 13 nodes, 4 citations), and the
final report can carry [id] markers with no citations-map entry. CitationAgent
(gpt_researcher/skills/citation_verification.py) is wired into deep_research
only — the tree path (gpt_researcher/skills/tree_research.py) never verifies.

Contract pinned here (implemented in s2-impl):
  (a) after research_node, node.sources keeps only documents actually read
      (present in the node researcher's research_sources) AND quoted in the
      node answer; a retriever-returned-but-unused URL and a read-but-unquoted
      document do not survive.
  (b) find_uncited_ids(report_md, citation_map) returns the [id] markers with
      no citation-map entry; run() detects them (result["uncited_ids"], list of
      str ids) and fail-closed strips them from the final report_md.
  (c) run() executes CitationAgent verification over the tree node claims
      (learnings mapped to their node's source URL) and exposes the
      CitationAgent.verify result as result["citation_verification"], with
      claims no read document supports flagged verified=False.

Deterministic, no network: GPTResearcher and create_chat_completion are
patched at the tree_research module seam (its docstring names that seam);
CitationAgent's network fetch is patched at CitationAgent._scrape.
"""
from types import SimpleNamespace

URL_QUOTED = "https://quoted.example.com/doc"
URL_READ_UNQUOTED = "https://read-unquoted.example.com/doc"
URL_RETRIEVER_ONLY = "https://retriever-only.example.com/doc"

QUOTED_DOC = (
    "The quick brown fox jumps over the lazy dog in 2024. "
    "Extra surrounding prose the answer never uses."
)
# deliberately shares no significant word with the answer below
UNQUOTED_DOC = "Basalt zeppelin marmalade cartography stalactite persimmon."

SUPPORTED_LEARNING = "The quick brown fox jumps over the lazy dog in 2024"
UNSUPPORTED_LEARNING = "Zorbulate framistan quuxly emitted gigawatt sprockets"

# one response serves every test: quotes QUOTED_DOC verbatim (keeps that source
# alive through narrowing), carries a fabricated [99] marker (uncited id), and
# yields one supported + one unsupported learning for CitationAgent.
LLM_RESPONSE = (
    f"ANSWER: {SUPPORTED_LEARNING}. A fabricated aside cited to nowhere [99].\n"
    f"DIGEST: {SUPPORTED_LEARNING} [99].\n"
    "LEARNINGS:\n"
    f"- {SUPPORTED_LEARNING}\n"
    f"- {UNSUPPORTED_LEARNING}\n"
)


def _make_skill(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.query = query
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            self.visited_urls.update(
                {URL_QUOTED, URL_READ_UNQUOTED, URL_RETRIEVER_ONLY})
            return "collected context"

        def get_research_sources(self):
            # documents actually read (scraped) — the retriever-only URL has none
            return [
                {"url": URL_QUOTED, "title": "quoted",
                 "raw_content": QUOTED_DOC},
                {"url": URL_READ_UNQUOTED, "title": "read-unquoted",
                 "raw_content": UNQUOTED_DOC},
            ]

        def get_costs(self):
            return 0.0

    async def fake_chat(*args, **kwargs):
        return LLM_RESPONSE

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    parent = SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    return tree_mod.TreeResearchSkill(parent)


async def _researched_node(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch)
    node = tree_mod.ResearchNode(id="0", question="root question",
                                 parent_id=None, depth=0)
    skill.nodes[node.id] = node
    await skill.research_node(node)
    return node


# ---------------------------------------------------------------------------
# (a) node.sources narrowed to documents actually read AND quoted
# ---------------------------------------------------------------------------

async def test_retriever_returned_but_unused_url_does_not_survive(monkeypatch):
    node = await _researched_node(monkeypatch)

    assert URL_RETRIEVER_ONLY not in node.sources, (
        "observed defect 6a: node.sources keeps every retriever-returned URL; "
        "a URL never scraped into the read documents must not survive"
    )
    # narrowing must keep the document the answer quotes, not empty the list
    assert URL_QUOTED in node.sources


async def test_read_but_unquoted_document_does_not_survive(monkeypatch):
    node = await _researched_node(monkeypatch)

    assert URL_READ_UNQUOTED not in node.sources, (
        "a document that was read but never quoted in the node answer must "
        "not survive — node.sources means 'read AND quoted'"
    )
    assert URL_QUOTED in node.sources


# ---------------------------------------------------------------------------
# (b) [id] markers with no citations-map entry are detected (uncited ids)
# ---------------------------------------------------------------------------

def test_find_uncited_ids_detects_ids_missing_from_citation_map():
    from gpt_researcher.skills.tree_research import find_uncited_ids

    report = ("Alpha claim [1]. Beta claim [2]. Gamma claim [7].\n\n"
              "## Citations\n\n- [1] https://a.example.com\n")
    assert sorted(find_uncited_ids(report, {"1": "https://a.example.com"})) == ["2", "7"]
    assert find_uncited_ids("all cited [1]", {"1": "https://a.example.com"}) == []


async def test_run_detects_and_strips_uncited_ids_fail_closed(monkeypatch):
    skill = _make_skill(monkeypatch)

    result = await skill.run(query="root question", max_depth=0, max_nodes=1)

    assert list(result["uncited_ids"]) == ["99"], (
        "run() must detect the [99] marker the LLM fabricated — it has no "
        "citations-map entry"
    )
    assert "[99]" not in result["report_md"], (
        "fail-closed: an uncited [id] must not survive into the final report"
    )


# ---------------------------------------------------------------------------
# (c) CitationAgent verification over tree node claims flags unverified
# ---------------------------------------------------------------------------

async def test_citation_agent_flags_unverified_tree_node_claims(monkeypatch):
    from gpt_researcher.skills.citation_verification import CitationAgent

    docs = {URL_QUOTED: QUOTED_DOC}
    monkeypatch.setattr(CitationAgent, "_scrape",
                        lambda self, url: docs.get(url))

    skill = _make_skill(monkeypatch)
    result = await skill.run(query="root question", max_depth=0, max_nodes=1)

    cv = result["citation_verification"]
    assert cv["total_claims"] >= 2, (
        "tree node learnings must be fed to CitationAgent as claims"
    )
    quotes = [c["quote"] for c in cv["claims"]]
    assert any(SUPPORTED_LEARNING.lower() in q.lower() for q in quotes)
    unverified = [c for c in cv["claims"] if not c["verified"]]
    assert any("zorbulate" in c["quote"].lower() for c in unverified), (
        "a claim no read document supports must be flagged verified=False"
    )
    assert cv["unverified"] >= 1
