"""Report readability: section titles and citation density.

Two measured defects on a real captured run
(outputs/graph-structured-engineering-of-llm-agen-a7756b9c.tree-report.md, 366
lines): 8 of 10 headings were comma-joined keyword bags
("## Emergent, excluded, manager, noting"), and citation markers clustered
mid-sentence ("[15] [17] [22] [24] [25]") at 1.25/line.

DEDUP-HARNESS.md logs the heading defect as R5 and leaves it undone there
because a prose title is a model call nothing in the s9 fixture can verify --
these tests hold the DETERMINISTIC replacement (a leading phrase quoted from
the theme's own representative unit) to that same no-model-call constraint.

The citation tests hold the density fix to the constraint that sank three
earlier citation-density rewrites in this repo (see _attribute_citations's
docstring): a relocated marker must still trace to its source inside the
frozen scorer's own window. `test_cluster_relocation_preserves_grounding_*`
re-runs that exact check (phrase_traced, mirroring bench/score_report.py's
score_s1) against the shipped text rather than trusting the implementation.
"""
import gpt_researcher.skills.tree_research as tr
from gpt_researcher.skills.tree_research import TreeResearchSkill as T

# --------------------------------------------------------------- (a) titles


def test_title_is_not_a_comma_joined_keyword_bag():
    """R5's own example, reproduced: the old ranker joined single ranked words
    with commas ("Emergent, excluded, manager, noting"). The fix must not."""
    unit = ("**Burr** (official Apache Burr docs, burr.apache.org): Burr's own "
            "documentation [15] explicitly frames the framework around "
            "**deterministic graph routing**.")
    title = T._theme_title(unit, set())
    assert title == "Burr", title
    assert "," not in title


def test_title_is_literally_traceable_to_its_unit():
    """The point of R5's fix: a fixture can verify the title against real report
    text, because it IS real report text -- unlike an invented outline word."""
    unit = ("So both CrewAI Flows and [62] AutoGen GraphFlow self-describe their "
            "routing as deterministic, even though the content can be "
            "model-generated.")
    title = T._theme_title(unit, set())
    cleaned = tr._MD_NOISE_RE.sub("", tr._flat_claim(unit))
    # _theme_title only ever touches the FIRST character's case (to capitalize a
    # lowercase lead-in), so a case-insensitive substring check is the real quote
    # test -- an outline word invented from scratch would not pass this either.
    assert title.lower() in cleaned.lower(), f"{title!r} is not a real quote of {cleaned!r}"


def test_title_never_ships_an_orphan_delimiter():
    """A unit opening on a full quotation or a markdown aside must not leave a
    dangling closer/opener behind when the lead phrase is cut short."""
    quoted = '"A separate table stores the compensating actions" [9], per the docs.'
    aside = ("**Burr** (official Apache Burr docs, burr.apache.org and the "
             "GitHub .rst source) explicitly frames routing as deterministic.")
    for unit in (quoted, aside):
        title = T._theme_title(unit, set())
        assert title.count('"') % 2 == 0, title
        assert title.count("(") == title.count(")"), title
        assert not title.endswith((",", ";", ":")), title


def test_duplicate_titles_are_disambiguated():
    """Two representative units must never ship the SAME H2 -- that reads as one
    section split in half, the same guarantee the old keyword-bag title made."""
    used: set = set()
    a = T._theme_title("The gate blocks a transition until a check passes.", used)
    b = T._theme_title("The gate blocks a transition until a check passes, too.", used)
    assert a != b
    assert {a, b} <= used


def test_empty_unit_falls_back_to_findings_and_still_disambiguates():
    used: set = set()
    only_markers = T._theme_title("[3] [4]", used)
    blank = T._theme_title("   ", used)
    assert only_markers == "Findings"
    assert blank != only_markers


# ----------------------------------------------------------- (b) citations


def test_repeated_citation_in_one_paragraph_collapses_to_its_last_occurrence():
    """Measured on the real report: id 35 was cited twice in one sentence,
    "[10] [35] [36] are the same [32] problem ... designed for [35],". Dropping
    the earlier copy needs no re-verification -- both occurrences were already
    independently earned before this pass ever sees them."""
    para = ('Noting "long-running [10] [35] [36] are the same [32] problem '
            'Temporal was designed for [35]," explicitly framing this as commentary.')
    out = T._drop_repeat_cites(para)
    assert out.count("[35]") == 1, out
    assert out.index("[35]") > out.index("[32]"), "must keep the LAST occurrence"
    for cid in ("10", "36", "32"):
        assert out.count(f"[{cid}]") == 1, f"an id cited once must not be touched: {cid}"


def test_cluster_relocation_preserves_grounding_measured_the_scorers_way():
    """Not an argument that the code is careful -- literally re-run the scorer's
    own check (phrase_traced) against the text as SHIPPED, at each marker's new
    position. This is the decisive measurement the trap in this file's docstring
    demands: three earlier rewrites broke exactly this without checking it."""
    docs = {cid: "our system uses a hybrid retrieval approach"
            for cid in ("15", "17", "22")}
    para = ("Our system uses a hybrid retrieval approach [15] [17] [22] that "
            "combines dense and sparse vectors for ranking.")
    out = T._loosen_cluster(para, docs)

    assert out != para, "fixture guard: this case is supposed to relocate"
    assert "that combines dense" in out.split("[15]")[0], (
        "the cluster must have moved past the clause it was interrupting"
    )
    for cid, doc in docs.items():
        marker = f"[{cid}]"
        pos = out.index(marker)
        window = out[:pos]
        assert tr.phrase_traced(window, doc), (
            f"marker {marker} no longer grounds at its new position: {out!r}"
        )


def test_cluster_stays_put_when_relocation_would_lose_grounding():
    """Same cluster, but the sentence is long enough that the traced phrase would
    fall outside the scorer's 240-char / 20-token window if moved to the end --
    the exact failure mode that cost three earlier rewrites their citations."""
    docs = {cid: "our system uses a hybrid retrieval approach"
            for cid in ("15", "17", "22")}
    filler = (" extra descriptive padding words here to lengthen this sentence "
              "significantly") * 4
    para = (f"Our system uses a hybrid retrieval approach [15] [17] [22] that "
            f"combines dense and sparse vectors{filler}.")
    out = T._loosen_cluster(para, docs)
    assert out == para, "an unsafe relocation must never ship"


def test_pair_of_markers_is_not_treated_as_a_cluster():
    """R5's complaint is a RUN of 3+ markers interrupting a sentence. A normal
    two-source citation is not that, and must be left exactly as attributed."""
    para = "The API returns a 429 on quota exhaustion [1] [2] according to the docs."
    out = T._loosen_cluster(para, {"1": "quota exhaustion docs", "2": "quota exhaustion docs"})
    assert out == para


def test_relocation_never_lands_inside_a_decimal_or_identifier():
    """Measured live on the real captured report: a naive `[.!?]` search reads a
    decimal point as a sentence end, and moved a cluster INTO an arXiv id --
    "arXiv:2310.03714 [48]" (before) became "arXiv:2310 [2] [3].03714 [4] [48]"
    (after a first cut of this fix). _loosen_cluster must use the module's own
    sentence-boundary rule (_SENT_BREAK_RE), which requires whitespace after the
    punctuation and so never fires inside a number."""
    docs = {cid: "text about the DSPy paper Khattab et al 2023" for cid in ("2", "3")}
    para = ("The base DSPy paper (Khattab et al., 2023 [2] [3] [37], "
            "arXiv:2310.03714 [48]) frames DSPy programs as text transformation graphs.")
    out = T._loosen_cluster(para, docs)
    assert "2310.03714" in out, out


def test_cluster_already_at_sentence_end_is_left_alone():
    """Nothing to gain by moving a cluster that already sits at the end of its
    own sentence -- and _loosen_cluster must not touch it."""
    docs = {cid: "text" for cid in ("1", "2", "3")}
    para = "This claim is well supported by three independent sources [1] [2] [3]."
    out = T._loosen_cluster(para, docs)
    assert out == para
