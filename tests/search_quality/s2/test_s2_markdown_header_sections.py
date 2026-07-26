"""Regression guard — the model may label ANSWER/DIGEST/LEARNINGS sections with
markdown headers ("# ANSWER") instead of the requested "ANSWER:" prefix. The
old regex only matched the colon form, so parsing silently failed end-to-end
and node.answer_md fell back to response.strip() -- the ENTIRE reply (all
three sections, headers included) dumped as one blob. That fed
_attribute_citations paraphrased DIGEST/LEARNINGS text it can't ground as
tightly as the real ANSWER wording, which is what depressed a live S1_pct
measurement to 48%/63% (confirmed: the rendered report.md had "# DIGEST" and
"# LEARNINGS" literally glued mid-sentence into the body). Deterministic, no
network: exercises research_node's response parsing directly.
"""
from types import SimpleNamespace

URL = "https://quoted.example.com/doc"
DOC = "The team ported the codebase to Rust in eleven days during 2024."

ANSWER_TEXT = "The team ported the codebase to Rust in eleven days during 2024 [1]."
DIGEST_TEXT = "A short paraphrase of the port."
LEARNING_1 = "The port took eleven days."
LEARNING_2 = "Rust was the target language."

MARKDOWN_HEADER_RESPONSE = (
    f"# ANSWER\n\n{ANSWER_TEXT}\n\n"
    f"# DIGEST\n\n{DIGEST_TEXT}\n\n"
    f"# LEARNINGS\n\n- {LEARNING_1}\n- {LEARNING_2}\n"
)


def _make_skill(monkeypatch, response: str):
    import gpt_researcher.skills.tree_research as tree_mod

    class _FakeNodeResearcher:
        def __init__(self, query=None, visited_urls=None, **kwargs):
            self.visited_urls = visited_urls if visited_urls is not None else set()

        async def conduct_research(self):
            self.visited_urls.add(URL)
            return "collected context"

        def get_research_sources(self):
            return [{"url": URL, "title": "quoted", "raw_content": DOC}]

        def get_costs(self):
            return 0.0

    async def fake_chat(*args, **kwargs):
        return response

    monkeypatch.setattr(tree_mod, "GPTResearcher", _FakeNodeResearcher)
    monkeypatch.setattr(tree_mod, "create_chat_completion", fake_chat)

    parent = SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    )
    return tree_mod.TreeResearchSkill(parent)


async def test_markdown_header_sections_are_split_not_dumped_whole(monkeypatch):
    import gpt_researcher.skills.tree_research as tree_mod

    skill = _make_skill(monkeypatch, MARKDOWN_HEADER_RESPONSE)
    node = tree_mod.ResearchNode(id="0", question="root question",
                                 parent_id=None, depth=0)
    skill.nodes[node.id] = node
    await skill.research_node(node)

    assert node.answer_md.strip() == ANSWER_TEXT, (
        "a '# ANSWER' markdown header must parse the same as 'ANSWER:' -- "
        f"got: {node.answer_md!r}"
    )
    assert "# DIGEST" not in node.answer_md and "# LEARNINGS" not in node.answer_md, (
        "on parse failure the whole reply (headers included) falls back into "
        "answer_md -- it must never carry the other sections' headers"
    )
    assert node.answer_digest.strip() == DIGEST_TEXT
    assert node.learnings == [LEARNING_1, LEARNING_2]
