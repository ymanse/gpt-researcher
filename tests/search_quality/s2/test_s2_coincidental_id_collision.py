"""Regression guard — a raw LLM-written [id] that coincidentally matches a REAL
citation_map entry (just not the source it sits next to) must not survive.

Root cause (measured live, bun-rust-port + outbox-failure-modes s2 measure
runs): node.answer_md is the answer LLM's own markdown, and research_node's
prompt used to ask for "a cited markdown answer" — the model invents its own
[N] numbers with no knowledge of citation_map's real global numbering. When
the tree has enough sources that citation_map spans a wide id range, the
model's invented number often lands ON a real id by pure coincidence, so
find_uncited_ids (which only catches ids with NO citation_map entry at all,
see test_s2_citation_integrity.py) cannot flag it — the id is valid, just for
an unrelated source. Observed live: `[13]` sitting next to a sentence about
US engineer salaries while citation_map's real #13 was an OpenTelemetry
tracing article, and whole clusters like `[4] [8] [10] [11] [12] [14] [16]
[17]` dumped after a numbered-list marker with no per-claim relationship to
any of them.

_prune_ungrounded_markers is the second fail-closed pass (after
find_uncited_ids) that catches this: every surviving marker's own sentence is
re-checked against that SPECIFIC id's read document, independent of who wrote
the marker.
"""
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod


def _skill() -> tree_mod.TreeResearchSkill:
    return tree_mod.TreeResearchSkill(SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    ))


async def test_coincidentally_valid_id_not_supporting_its_sentence_is_pruned():
    skill = _skill()
    skill._read_docs["https://a.example.com"] = "unrelated filler content about gardening tools"
    skill._read_docs["https://b.example.com"] = "the quick brown fox jumps over the lazy dog in 2024"
    citation_map = {"1": "https://a.example.com", "2": "https://b.example.com"}
    body = "The quick brown fox jumps over the lazy dog in 2024 [1]."

    pruned = skill._prune_ungrounded_markers(body, citation_map)

    assert "[1]" not in pruned, (
        "id 1 is a genuine citation_map entry, so find_uncited_ids lets it "
        "through — but source #1's own document never supported this "
        "sentence (source #2's would have), so it must still be dropped"
    )


async def test_genuinely_grounded_marker_survives_the_prune():
    skill = _skill()
    skill._read_docs["https://a.example.com"] = "the quick brown fox jumps over the lazy dog in 2024"
    citation_map = {"1": "https://a.example.com"}
    body = "The quick brown fox jumps over the lazy dog in 2024 [1]."

    pruned = skill._prune_ungrounded_markers(body, citation_map)

    assert "[1]" in pruned, "a marker whose own source really does support its sentence must survive"


async def test_cluster_of_unrelated_ids_after_a_list_marker_is_pruned():
    """The live-observed "2. [4] [8] [10] [11]" pattern: several real ids dumped
    together next to a fragment none of them actually supports."""
    skill = _skill()
    for n, text in [(4, "alpha beta gamma document"), (8, "delta epsilon zeta document")]:
        skill._read_docs[f"https://{n}.example.com"] = text
    citation_map = {"4": "https://4.example.com", "8": "https://8.example.com"}
    body = "2. [4] [8]"

    pruned = skill._prune_ungrounded_markers(body, citation_map)

    assert "[4]" not in pruned and "[8]" not in pruned, (
        "a numbered-list marker fragment supports neither real source; both "
        "clustered ids must be dropped even though both exist in citation_map"
    )
