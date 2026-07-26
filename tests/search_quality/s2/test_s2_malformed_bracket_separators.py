"""Regression guard — review R1: malformed bracket separators must not escape the
fail-closed uncited-[id] strip.

_CITE_ID_RE required the ENTIRE bracket interior to match item(sep item)* exactly, so
a single malformed separator (a trailing comma before "]", or a doubled separator with
no item between) made the whole bracket invisible to both find_uncited_ids and the
_CITE_ID_RE.sub strip -- a fabricated/uncited id inside it survived into report_md
untouched and uncounted. _SEP now tolerates repeated punctuation and an optional
dangling separator before the closing bracket; a genuine date range like
"[2024-07-26]" must still be excluded (a hyphen is never a valid separator character
outside _ITEM's own range form).
"""
from gpt_researcher.skills.tree_research import _bracket_ids, find_uncited_ids

CITED = {"1": "https://a.example.com", "2": "https://b.example.com"}


def test_trailing_comma_before_closing_bracket_is_still_detected():
    report = "Alpha claim [1, 99,]."
    assert find_uncited_ids(report, CITED) == ["99"]


def test_doubled_separator_between_ids_is_still_detected():
    report = "Alpha claim [1,2,,3]."
    assert find_uncited_ids(report, CITED) == ["3"]


def test_bracketed_date_is_still_not_mistaken_for_a_citation():
    report = "Published [2024-07-26] with no citations at all."
    assert find_uncited_ids(report, {}) == []


def test_bracket_ids_strips_trailing_and_doubled_separator_noise():
    assert _bracket_ids("1, 99,") == ["1", "99"]
    assert _bracket_ids("1,2,,3") == ["1", "2", "3"]
