"""A list marker is not a claim — an enumerated list must not shred into empty numbers.

Measured 2026-08-04 on a shipped deep_tree_research report (the harness-landscape run,
`graph-structured-engineering-of-llm-agen-a7756b9c`): **8 of 139 claim units (5.8%)**
were bare ordinals — `**1.` `**2.` `**3.` `1.` `2.` `3.` `4.` `5.` — and the rendered
report carried

    **Key contrasts:**

    1.

    2.

    3.

with the five actual items filed under other headings, because every unit is laid out
by theme and a number carries no topic to be laid out by.

The cause is that `1. Academic representations…` satisfies every clause of
`_SENT_BREAK_RE`: a full stop, whitespace, then a capital. Nothing distinguishes it
from the end of a sentence. `\\s+` in that pattern already protects a decimal point
("3.5"), but a list marker has the space.

The remedy reuses the rule that already exists for a colon lead-in: a marker governs
what follows, so it travels with it. What is pinned here is the OUTCOME — no claim unit
is a bare marker, and the item text stays attached to its own number — not the
mechanism, which is free to change.
"""
from __future__ import annotations

import re

from gpt_researcher.skills.tree_research import _claim_units

BARE_MARKER = re.compile(r"^[\s*_>#-]*(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})\s*[.)]\s*$")

NUMBERED = (
    "**Key contrasts:**\n\n"
    "1. Academic work represents the build as an explicit graph whose edges a "
    "deterministic router picks.\n\n"
    "2. Verifier-in-the-loop gating advances a stage only on tool-emitted evidence.\n\n"
    "3. Ralph-loop iteration re-invokes the agent until a completion predicate holds.\n"
)

MARKDOWN_HEADS = (
    "**1. Deterministic routing**\n\n"
    "Conductor declares topology in YAML and resolves edges with first-match-wins.\n\n"
    "**2. Fail-closed gating**\n\n"
    "A stage advances only when a checker validates the artifact the tool emitted.\n"
)

PARENS = (
    "Three defensible patterns:\n\n"
    "a) Adjacency lists keep writes cheap.\n\n"
    "b) Closure tables keep deep reads cheap.\n"
)


def _units(text):
    return _claim_units(text)


def test_no_claim_unit_is_a_bare_list_marker():
    for label, text in (("numbered", NUMBERED), ("bold heads", MARKDOWN_HEADS),
                        ("lettered", PARENS)):
        bare = [u for u in _units(text) if BARE_MARKER.match(u)]
        assert not bare, f"{label}: a number is not a claim, yet these units are only that: {bare}"


def test_each_item_stays_attached_to_its_own_number():
    units = _units(NUMBERED)
    for n, needle in ((1, "deterministic router"), (2, "tool-emitted evidence"),
                      (3, "completion predicate")):
        owner = [u for u in units if needle in u]
        assert owner, f"item {n} vanished entirely: {units}"
        assert re.search(rf"(?<!\d){n}\s*[.)]", owner[0]), (
            f"item {n} lost its number — it will be laid out by theme with nothing to "
            f"say which item it was: {owner[0][:120]!r}"
        )


def test_a_bold_numbered_heading_keeps_its_heading_text():
    units = _units(MARKDOWN_HEADS)
    heads = [u for u in units if "Deterministic routing" in u]
    assert heads, f"the heading text was lost: {units}"
    assert heads[0].lstrip().startswith(("**1", "1")), (
        f"the heading was severed from its number: {heads[0][:100]!r}"
    )


def test_ordinary_sentences_still_split():
    """The guard must not swallow real sentence boundaries — that would undo the whole
    point of an atomic claim unit (a paragraph carrying five facts merges nothing)."""
    prose = (
        "Postgres can get you far. Graphile Worker processes 180,000 jobs per second. "
        "That is fifteen billion jobs per day."
    )
    units = _units(prose)
    assert len(units) >= 2, f"a three-sentence paragraph must not stay one unit: {units}"


def test_a_decimal_is_still_not_a_break():
    """Pre-existing behaviour this change must not disturb."""
    units = _units("The median within-model range is 13.6 percent across harnesses.")
    assert len(units) == 1, f"a decimal point is not a sentence end: {units}"
