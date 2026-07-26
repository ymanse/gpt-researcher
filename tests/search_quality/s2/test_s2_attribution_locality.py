"""s2 impl regression — review R1: word overlap must localize to one passage.

A topically-adjacent document sharing the claim's vocabulary scattered
document-wide is NOT a quoted source; only verbatim containment or a single
passage concentrating the claim's significant words counts as support.
Deterministic, no network: exercises text_supported directly.
"""
from gpt_researcher.skills.citation_verification import text_supported

CLAIM = "Bun ported 530k lines of Zig code to Rust in nine days using agents"

_CLAIM_WORDS = ["ported", "530k", "lines", "code", "rust",
                "nine", "days", "using", "agents"]

# every significant claim word appears, but no two within one passage —
# topical co-occurrence across an article, not a supporting passage
_FILLER = " ".join(["filler"] * 80)
SCATTERED_DOC = _FILLER.join(f" {w} " for w in _CLAIM_WORDS)

# the same vocabulary concentrated in one paraphrased passage
PASSAGE_DOC = (
    " ".join(["intro"] * 100)
    + " In practice Bun ported 530k lines of Zig code to Rust in nine "
    "days using coding agents. "
    + " ".join(["outro"] * 100)
)


def test_scattered_topical_overlap_is_not_quoted():
    assert not text_supported(CLAIM, SCATTERED_DOC), (
        "review R1: document-wide word scatter is topical similarity, not "
        "quotation — it must not keep a source alive through narrowing"
    )


def test_concentrated_passage_supports_paraphrase():
    assert text_supported(CLAIM, PASSAGE_DOC)


def test_verbatim_containment_still_supported():
    assert text_supported("nine days", "the rewrite took nine days total")
