"""RED test for s4-measure — a verbatim quote diluted by surrounding exposition
(spec/search-quality.md, observed on the live s4 measure: bun-rust-port / github.com
and outbox-failure-modes / microservices.io both landed S4_pct=50, one required
domain short, even though the node that read each domain's canonical page WAS
researched (frontier priority worked)).

Observed: research_node splits node.answer_md on sentence boundaries
(`re.split(r"(?<=[.!?])\\s+", ...)`) before checking each chunk with
text_supported(). That regex requires the sentence-ending punctuation to sit
directly before the whitespace -- a quoted sentence ending `..."` (period THEN
closing quote THEN space) does not match, so the quoted sentence never splits
from the exposition that follows it in the same bullet. The merged chunk then
carries many claim words the source page never contains (the answer LLM's own
connective prose), which drags _passage_covers's 70%-of-words-in-one-window
ratio below threshold even though the quote itself is grounded almost verbatim.

Contract pinned here: quoted spans in node.answer_md are checked standalone (in
addition to the sentence-split chunks), so a genuine verbatim quote grounds its
source even when the sentence carrying it also carries invented exposition.

Deterministic, no network: GPTResearcher and create_chat_completion are patched
at the tree_research module seam, matching tests/search_quality/s2/test_s2_citation_integrity.py.
"""
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod

URL_DOC = "https://microservices.io/patterns/data/transactional-outbox.html"

QUOTE = ("potentially error prone since the developer might forget to publish "
         "the message after updating the database")
DOC = (f'The pattern is documented as "{QUOTE}" in the resulting context section. '
       "Unrelated archive filler text padding out the rest of the page. " * 5)

# the exposition clause shares no real word with DOC -- pure invented connective
# prose, same shape as the live answer's "Because the outbox write is a manual,
# separate insert..." clause that followed its own verbatim quote
NONSENSE_EXPOSITION = ("Zorbulate framistan quuxly emitted gigawatt sprockets while "
                       "wobulated quantifier teams remained vigilant about recurring "
                       "failure modes across distributed deployments")

LLM_RESPONSE = (
    f'ANSWER: The pattern is "{QUOTE}." {NONSENSE_EXPOSITION} explains the risk.\n'
    f'DIGEST: The pattern is "{QUOTE}."\n'
    "LEARNINGS:\n"
    f"- The pattern is \"{QUOTE}.\" {NONSENSE_EXPOSITION} explains the risk.\n"
)


def _make_skill(monkeypatch):
    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            self.visited_urls.add(URL_DOC)
            return "Collected page text from the scraped sources. " * 300  # >> MIN_CONTEXT_CHARS

        def get_research_sources(self):
            return [{"url": URL_DOC, "title": "doc", "raw_content": DOC}]

        def get_costs(self):
            return 0.0

    async def fake_chat(*args, **kwargs):
        return LLM_RESPONSE

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    parent = SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    )
    return tree_mod.TreeResearchSkill(parent)


async def test_quote_diluted_by_exposition_still_grounds_its_source(monkeypatch):
    skill = _make_skill(monkeypatch)
    node = tree_mod.ResearchNode(id="0", question="root question", parent_id=None, depth=0)
    skill.nodes[node.id] = node

    await skill.research_node(node)

    assert URL_DOC in node.sources, (
        "a verbatim quote embedded in a sentence that also carries invented "
        "exposition must still ground its source -- checking the whole "
        "sentence as one claim dilutes the quote's word-overlap below the "
        "70% passage threshold even though the quote itself is near-verbatim"
    )
