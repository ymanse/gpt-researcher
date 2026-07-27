"""Regression guard — review R1: what a fail-closed pass KEEPS must stay readable
by the frozen S1 scorer.

Both marker passes parse a bracket's ids with _bracket_ids (which understands
combined groups like "[7, 9]") and then re-render the survivors. Re-rendering
them as one comma-joined bracket makes every kept id invisible to
bench/score_report.py's CITE regex, which matches only digits sitting directly
between the brackets — so an id the pass deliberately protected still lands in
citations_total (through its "- [id] url" line in the Citations block) but can
never land in citations_grounded. The strip meant to protect S1_pct lowers it.

Checked against the frozen scorer's own regex, loaded from bench/, not a copy.

Deterministic, no network.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tree_mod

_SCORER = pathlib.Path(__file__).resolve().parents[3] / "harness-search" / "bench" / "score_report.py"
_spec = importlib.util.spec_from_file_location("frozen_score_report", _SCORER)
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)

URL_A, URL_B = "https://a.example.com/doc", "https://b.example.com/doc"
PHRASE = "the outbox table grows past a hundred million rows"
DOC = " ".join(["unrelated"] * 50) + " " + PHRASE + " " + " ".join(["filler"] * 50)


def _skill() -> tree_mod.TreeResearchSkill:
    return tree_mod.TreeResearchSkill(SimpleNamespace(
        query="root question",
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set(),
    ))


def _scorer_ids(text: str) -> list:
    return [m.group(1) for m in frozen.CITE.finditer(text)]


def test_comma_joined_survivors_would_be_invisible_to_the_scorer():
    """Pins the premise: the shape the passes used to write grounds nothing."""
    assert _scorer_ids("... rows [7, 9] ...") == []


def test_two_survivors_of_one_bracket_stay_countable():
    assert _scorer_ids(tree_mod.render_ids(["7", "9"])) == ["7", "9"]


def test_uncited_strip_leaves_the_survivor_readable():
    kept = tree_mod._bracket_ids("7, 99")
    assert _scorer_ids(tree_mod.render_ids([c for c in kept if c != "99"])) == ["7"]


def test_ungrounded_pass_keeps_both_grounded_ids_countable():
    skill = _skill()
    skill._read_docs[URL_A] = skill._read_docs[URL_B] = DOC

    body = skill._prune_ungrounded_markers(f"{PHRASE} [7, 9].", {"7": URL_A, "9": URL_B})

    assert _scorer_ids(body) == ["7", "9"], (
        "both ids traced to their source and were kept — a grader that cannot "
        "see them counts them in the denominator only"
    )
