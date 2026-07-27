"""Regression guard — the fail-closed marker passes must judge locality by the
same rule the frozen S1 scorer does.

Root cause (measured, round-1 benchmark): both marker passes asked
text_supported — ">=70% of the span's significant words inside one 60-token
window" — while bench/score_report.py's score_s1 asks something far weaker and
in a different dimension: does any 3-token window of the last 20 normalized
tokens before the marker appear VERBATIM in the cited page. A near-verbatim
quote surrounded by the answer LLM's own exposition clears the scorer and fails
the coverage rule, so the passes stripped markers the grader would have counted.
Two of the five golden reports (edge-ai-face-access, outbox-failure-modes) came
out with 25 and 55 citation-map entries and ZERO surviving markers, and score_s1
scores an id-less report 0.0 — S1 mean 42 against an 80 baseline.

Deterministic, no network.
"""
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod
from gpt_researcher.skills.citation_verification import text_supported
from gpt_researcher.skills.tree_research import phrase_traced

URL = "https://source.example.com/doc"

# the answer's wording: one verbatim borrowing from the source, everything else
# the model's own connective prose
CLAIM = ("Operators reported that the outbox table grows past a hundred million "
         "rows before anyone opens the dashboard, which is when the pager finally "
         "goes off and somebody starts asking about retention policy")
# the borrowed phrase really is in the source, but the source spends its length
# on unrelated material, so the claim's vocabulary never concentrates
DOC = (" ".join(["unrelated"] * 200)
       + " the outbox table grows past a hundred million rows "
       + " ".join(["filler"] * 200))


def _skill() -> tree_mod.TreeResearchSkill:
    return tree_mod.TreeResearchSkill(SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    ))


def test_sentence_end_placement_does_not_ground_an_earlier_borrowing():
    """Pins both halves of the premise: the old passes rejected this span for
    lacking passage coverage, and the old PLACEMENT (marker after the final
    word) puts the borrowed phrase outside the 20 tokens the grader reads."""
    assert not text_supported(CLAIM, DOC), (
        "the passage-coverage rule rejects a verbatim borrowing diluted by "
        "exposition — this is the span the old passes threw away"
    )
    assert not phrase_traced(CLAIM, DOC), (
        "20 tokens of exposition follow the borrowing, so a marker at the "
        "sentence end sees none of it either"
    )


def test_marker_lands_on_the_traced_phrase_and_survives_its_own_audit():
    skill = _skill()
    skill._read_docs[URL] = DOC

    attributed = skill._attribute_citations(f"{CLAIM}.", {URL: "1"})

    assert "[1]" in attributed, "a verbatim borrowing must earn its source a marker"
    assert attributed.index("[1]") < attributed.index("before anyone"), (
        "the marker belongs on the borrowed phrase; at the sentence end the "
        "grader reads 20 tokens of the model's own exposition instead"
    )
    assert "[1]" in skill._prune_ungrounded_markers(attributed, {"1": URL}), (
        "placement and audit must agree — a marker the fail-closed pass strips "
        "is one the grader would have counted, and stripping every one of them "
        "scores the report 0.0"
    )


def test_stopword_only_window_traces_nothing():
    """The scorer's own guard: a 3-token window needs a 4+ char token, so list
    numbering and function words never ground a marker by coincidence."""
    assert not phrase_traced("2. and of to", "2. and of to a b c")


def test_paraphrase_only_source_earns_no_marker():
    """A source kept by node.sources narrowing on passage coverage alone adds an
    id that can never ground — it would only enlarge score_s1's denominator."""
    skill = _skill()
    paraphrase_doc = ("The team completed the port of the codebase over to Rust, "
                      "taking eleven days across 2024.")
    skill._read_docs[URL] = paraphrase_doc
    sentence = "The codebase port to Rust took eleven days in 2024."

    assert text_supported(sentence, paraphrase_doc), "the source does support the node"
    assert "[1]" not in skill._attribute_citations(sentence, {URL: "1"}), (
        "support without a verbatim trace is not a place to put a marker"
    )
