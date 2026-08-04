"""Stage 3 — "I could not read the source" and "the source does not say this" are
different findings, and the claim record must say which.

Measured 2026-08-04. A research report quoted a figure from preprints.org, cross-checked
it, found no support, and DISCARDED IT AS WRONG. The figure was right — the sentence
"the median within-model range is 13.6% and 14 of the 20 models vary by at least 10%
across harnesses" is in the paper verbatim. What had happened is that preprints.org
served our scraper an Akamai interstitial (HTTP 200, 32 characters of text), the page was
dropped as "content too short", and the only text about that source left in the run was
its abstract — which contains no figures at all. The check then compared a full-text
claim against an abstract and reported the source as disagreeing.

`verified: False` carried both meanings, so nothing downstream could tell a hole in the
evidence from evidence against the claim. Only one of those is a reason to delete a
finding; the other is a reason to fetch the page again.

WHY A PER-CLAIM FLAG AND NOT A THIRD COUNTER. tests/tier_a/stage3/test_stage3_citation.py
is hash-pinned and fixes the shape of this result: :126 requires an unfetchable url to
report `unverified == 1` with `verified is False`, :151 requires
`total_claims == grounded + unverified`, and :157 asserts EXACT dict equality on the
empty result — so no new top-level key may appear and unretrieved claims must stay
counted inside `unverified`. The distinction therefore lives on the claim, as a
sub-partition of what is already there. These tests pin that it is present AND that the
old arithmetic still holds.

Deterministic, no network: the fetch seam is a tripwire.
"""
from unittest import mock

import pytest

from gpt_researcher.skills.citation_verification import CitationAgent

QUOTE = "the median within-model range is 13.6% across harnesses"
BLOCKED_URL = "https://www.preprints.org/manuscript/202606.1312/v1"
READ_URL = "https://example.test/read-but-silent"
# A page we DID read, about the right topic, that simply does not carry the figure.
READ_PAGE = (
    "# Agent harness design\n\nThe survey decomposes the execution harness into six "
    "runtime responsibilities: observation, context, control, action, state and "
    "verification. It argues that agent quality emerges from the interaction between "
    "model capability and runtime infrastructure.\n"
)


@pytest.fixture(autouse=True)
def firecrawl_key(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")


def _verify(citations, documents):
    """CitationAgent over a fixed corpus, with the network wired to explode."""
    with mock.patch(
        "requests.post",
        side_effect=AssertionError("no test here may reach the network"),
    ):
        return CitationAgent().verify(citations, documents=documents)


def test_a_source_we_never_read_is_marked_unretrieved():
    """The defect, stated as an outcome: unverified, and visibly for want of the page."""
    result = _verify({QUOTE: BLOCKED_URL}, {})

    claim = result["claims"][0]
    assert claim["verified"] is False
    assert claim.get("unretrieved") is True, (
        "a claim whose source was never obtained must say so on the claim — otherwise "
        "it is indistinguishable from one the source refutes, and gets deleted as wrong"
    )


def test_a_source_we_did_read_that_lacks_the_quote_is_not_marked_unretrieved():
    """The other direction, and the one that makes the flag mean anything. This page WAS
    read; it genuinely does not support the quote. That is evidence, not a hole."""
    result = _verify({QUOTE: READ_URL}, {READ_URL: READ_PAGE})

    claim = result["claims"][0]
    assert claim["verified"] is False
    assert "unretrieved" not in claim, (
        "we hold this page; failing to find the quote in it is a finding about the "
        f"CLAIM, not about our retrieval: {claim}"
    )


def test_the_two_are_distinguishable_in_one_run():
    """Both failures in a single result set, telling apart the report's exact situation:
    a correct claim from a walled-off source, beside a claim its source really lacks."""
    result = _verify(
        {QUOTE: BLOCKED_URL, "some other assertion entirely": READ_URL},
        {READ_URL: READ_PAGE},
    )

    by_url = {c["url"]: c for c in result["claims"]}
    assert by_url[BLOCKED_URL].get("unretrieved") is True
    assert "unretrieved" not in by_url[READ_URL]
    assert result["unverified"] == 2, "both are still unverified — this adds detail, not leniency"


def test_the_pinned_arithmetic_is_unchanged():
    """The sub-partition may not disturb the buckets test_stage3_citation.py pins."""
    result = _verify(
        {QUOTE: BLOCKED_URL, "another claim": READ_URL},
        {READ_URL: READ_PAGE},
    )

    assert result["total_claims"] == result["grounded"] + result["unverified"]
    assert result["grounded"] == sum(1 for c in result["claims"] if c["verified"])
    assert set(result) == {"total_claims", "grounded", "unverified", "claims"}, (
        "no new TOP-LEVEL key: test_stage3_citation.py:157 asserts exact dict equality "
        f"on the empty result, and that file is hash-pinned. Got {sorted(result)}"
    )


def test_an_empty_url_is_not_a_retrieval_failure():
    """The inversion that nearly shipped, and the tree path's dominant case.

    tree_research resolves each learning to a supporting source with
    `next((u for u in n.sources if text_supported(...)), "")` — so an EMPTY url means
    every source was read and none of them carried the claim. That is evidence against
    the claim, the exact opposite of a hole in the evidence. Flagging it `unretrieved`
    would have inverted the meaning in the path this fix matters most for.
    """
    result = _verify({QUOTE: ""}, {READ_URL: READ_PAGE})

    claim = result["claims"][0]
    assert claim["verified"] is False
    assert "unretrieved" not in claim, (
        "no url means the caller already searched what it holds and found nothing — "
        f"not that a fetch failed: {claim}"
    )


def test_a_grounded_claim_carries_no_flag():
    """A page we read that does support the quote stays exactly as it was."""
    page = "Analysis shows the median within-model range is 13.6% across harnesses.\n"
    result = _verify({QUOTE: READ_URL}, {READ_URL: page})

    claim = result["claims"][0]
    assert claim["verified"] is True
    assert "unretrieved" not in claim
    assert result["grounded"] == 1
