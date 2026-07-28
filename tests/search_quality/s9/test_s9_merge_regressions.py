"""s9 review round 1: the four ways the first merge deleted content, pinned.

test_s9_merge_contract.py pins the CONTRACT (same claim once, distinct claims
survive, groundings kept, no paste, titled sections). It is satisfied by a merge
that is still wrong in ways the fixture cannot see, and the review round found
four of them by replaying the rule over the captured corpus:

  R1  a figure the dropped statement carries and the survivor does not makes them
      different findings — two variants of one product, two sources' timelines
  R2  the identity test ran on _claim_profile, which discards tokens under four
      characters: "BYD", "LG", "SDI", "SK" were invisible, so two unexplored
      frontier questions differing only in the company they name merged into one
  R3  the separator after a merged sentence was discarded, and _scrubbed() blanks
      fenced code — so a code block sitting between two sentences went with it
  R4  rollup() handed _own_contribution an UNFILTERED summaries list while
      synthesize_node filtered falsy ones, so one FAILED leaf child made the
      parent's section swallow every surviving sibling's subtree

Deterministic like the contract file: assemble_report makes no LLM call and does
no retrieval, and these fixtures supply no documents at all, so nothing here can
reach the network.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tr

ROOT_Q = "What do the face-access and outbox corpora actually report?"


def _skill() -> tr.TreeResearchSkill:
    return tr.TreeResearchSkill(SimpleNamespace(
        query=ROOT_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock", strategic_llm_model="mock",
                            config_path=None),
        tone=None, websocket=None, headers={}, visited_urls=set()))


def _node(nid, question, *, learnings=(), answer="", depth=1,
          status=tr.NodeStatus.ANSWERED, children=()):
    n = tr.ResearchNode(id=nid, question=question,
                        parent_id=None if depth == 0 else nid.rsplit(".", 1)[0],
                        depth=depth)
    n.status = status
    n.learnings = list(learnings)
    n.answer_md = answer or "\n".join(learnings)
    n.answer_digest = n.answer_md[:120]
    n.sources = []
    n.children = list(children)
    return n


def _tree(*nodes) -> tr.TreeResearchSkill:
    skill = _skill()
    skill.nodes = {n.id: n for n in nodes}
    skill._read_docs = {}
    return skill


# --- R1 -------------------------------------------------------------------
V3 = ("The ASI7214Y-V3 datasheet states face verification accuracy above 99.5 percent "
      "with a comparison speed of 0.35 seconds per template.")
# quotes the V3 figure to name the discrepancy — which is exactly what gave the old
# symmetric rule a shared anchor to delete this sentence on
NON_V3 = ("The ASI7214Y non-V3 listing states a LOWER verification accuracy figure of "
          "above 99 percent, not above 99.5 percent, and a comparison speed no better "
          "than 0.55 seconds.")


async def test_a_figure_the_survivor_does_not_state_is_never_deleted():
    """R1: the deleted sentence was the one naming the discrepancy between two
    product variants — the merge kept the higher figure and dropped the lower."""
    report = (await _tree(
        _node("0", ROOT_Q, learnings=["Two datasheet revisions are in circulation."],
              depth=0, status=tr.NodeStatus.EXPANDED, children=["0.0", "0.1"]),
        _node("0.0", "What does the V3 datasheet state?", learnings=[V3]),
        _node("0.1", "What does the non-V3 listing state?", learnings=[NON_V3]),
    ).assemble_report(ROOT_Q))["report_md"]

    assert "99.5" in report and "0.35" in report, "the V3 figures were lost"
    assert "0.55" in report, (
        "the second variant's LOWER comparison speed was deleted as a duplicate of "
        "the first — differing figures are different findings, however alike the "
        f"wording:\n{report}")


LABEL = "**2026 pricing and licensing models**"
UNDER_LABEL = ("No 2026 pricing or licensing information for HID Amico, Suprema BioStar "
               "or Ajax Systems appears anywhere in the provided context.")


async def test_a_pseudo_heading_fragment_cannot_delete_the_finding_under_it():
    """R1/R6: a three-token bold label needed two shared tokens to swallow an
    eighteen-token finding, and first-wins then kept the label. The reader was
    left with a heading and no content."""
    report = (await _tree(
        _node("0", ROOT_Q, learnings=[LABEL], depth=0,
              status=tr.NodeStatus.EXPANDED, children=["0.0"]),
        _node("0.0", "What pricing is documented?", learnings=[UNDER_LABEL]),
    ).assemble_report(ROOT_Q))["report_md"]

    assert "HID Amico" in report and "Ajax Systems" in report, (
        f"the finding under the pseudo-heading was deleted:\n{report}")


# --- R2 -------------------------------------------------------------------
Q_BYD = ("What do BYD's own official announcements disclose about its in-house "
         "solid-state battery pilot production line, manufacturing bottlenecks and "
         "targeted mass-production timeline?")
Q_LG = ("What do LG's own official announcements disclose about its in-house "
        "solid-state battery pilot production line, manufacturing bottlenecks and "
        "targeted mass-production timeline?")


async def test_two_frontier_questions_naming_different_companies_both_survive():
    """R2: the only difference between these two is a token under four characters,
    which the scorer's profile discards — and a question is not a claim any other
    statement can be said to already make."""
    report = (await _tree(
        _node("0", ROOT_Q, learnings=["Two makers were queued and neither was reached."],
              depth=0, status=tr.NodeStatus.EXPANDED, children=["0.0", "0.1"]),
        _node("0.0", Q_BYD, status=tr.NodeStatus.PENDING),
        _node("0.1", Q_LG, status=tr.NodeStatus.PENDING),
    ).assemble_report(ROOT_Q))["report_md"]

    assert "BYD" in report, f"the BYD frontier question vanished:\n{report}"
    assert "LG" in report, f"the LG frontier question vanished:\n{report}"


# --- R3 -------------------------------------------------------------------
RICH = ("The unpublished outbox backlog passed 2,000,000 rows and the relay stalled "
        "completely across every shard in the payments cluster.")
WITH_FENCE = ("The unpublished outbox backlog passed 2,000,000 rows and the relay "
              "stalled.\n\n```sql\nSELECT id FROM outbox WHERE published_at IS NULL;\n"
              "```\n\nRecovery needed a manual replay.")


async def test_fenced_code_between_two_sentences_survives_the_merge():
    """R3: _scrubbed() blanks a fence to spaces, so a fence between two sentences
    lands INSIDE the separator. Dropping the separator with its merged sentence
    deleted the code out of the report."""
    report = (await _tree(
        _node("0", ROOT_Q, learnings=[RICH], depth=0,
              status=tr.NodeStatus.EXPANDED, children=["0.0"]),
        _node("0.0", "What did the relay incident review record?", answer=WITH_FENCE),
    ).assemble_report(ROOT_Q))["report_md"]

    assert report.count("2,000,000") == 1, (
        "precondition: the restatement must actually be merged away before asking "
        "what happened to the code block that followed it")
    assert "SELECT id FROM outbox WHERE published_at IS NULL;" in report, (
        f"the fenced query went to the grave with the merged sentence:\n{report}")


# --- R7 -------------------------------------------------------------------
CLAIM_A = ("The relay double-published 12,400 events in a single afternoon because two "
           "instances ran without a leader lock.")
CLAIM_B = ("Two relay instances ran without a leader lock and double-published 12,400 "
           "events.")


async def test_every_marker_the_merge_leaves_still_traces_to_its_own_page():
    """R7: the contract file's fixture appends every shared finding to EVERY source
    page, so a migrated [id] grounds there by construction and the "citations stay
    attached" condition passes for a reason that does not hold on real pages. Here
    each page carries only its own node's sentence, and the assertion is the
    invariant rather than a count: whatever markers survive the hand-off must each
    still trace to their own page by the grader's rule — the migration is not
    trusted, the fail-closed passes decide."""
    a = _node("0.0", "What did the payments postmortem record?", learnings=[CLAIM_A])
    b = _node("0.1", "What did the relay incident review record?", learnings=[CLAIM_B])
    a.sources, b.sources = ["https://example.test/a"], ["https://example.test/b"]
    skill = _tree(
        _node("0", ROOT_Q, learnings=["Two reviews covered the same incident."],
              depth=0, status=tr.NodeStatus.EXPANDED, children=["0.0", "0.1"]), a, b)
    skill._read_docs = {"https://example.test/a": CLAIM_A,
                        "https://example.test/b": CLAIM_B}

    result = await skill.assemble_report(ROOT_Q)
    body = result["report_md"].split("\n## Citations", 1)[0]

    assert body.count("12,400") == 1, f"precondition: one statement of the claim:\n{body}"
    markers = list(re.finditer(r"\[(\d{1,3})\](?!\()", body))
    assert markers, "vacuous: the merged claim carries no grounding at all"
    for m in markers:
        page = skill._read_docs[result["citation_map"][m.group(1)]]
        assert tr.phrase_traced(body[max(0, m.start() - 240):m.start()], page), (
            f"[{m.group(1)}] sits next to wording its own page never supported — a "
            f"marker the merge moved and nothing checked:\n{body}")


# --- R4 -------------------------------------------------------------------
GOOD_CHILD_Q = "What did the surviving sibling research?"
GOOD_CHILD = ("Change-data-capture connectors replaced the polling relay in 43 percent "
              "of the teams surveyed by the migration report.")


async def test_a_failed_leaf_child_does_not_swallow_its_siblings_section():
    """R4: a FAILED leaf returns "" from synthesize_node, so the unfiltered list
    _own_contribution was handed no longer matched the text it had to subtract —
    the parent kept the WHOLE roll-up and every child's section title vanished."""
    skill = _tree(
        _node("0", ROOT_Q, learnings=["Two branches were opened under the root."],
              depth=0, status=tr.NodeStatus.EXPANDED, children=["0.0", "0.1"]),
        _node("0.0", "What did the starved sibling research?", answer="",
              status=tr.NodeStatus.FAILED),
        _node("0.1", GOOD_CHILD_Q, learnings=[GOOD_CHILD]),
    )
    report = (await skill.assemble_report(ROOT_Q))["report_md"]

    assert report.count("43 percent") == 1, (
        f"the surviving child's finding was lost or duplicated:\n{report}")
    assert re.search(r"^#{2,6}\s+" + re.escape(GOOD_CHILD_Q) + r"\s*$", report, re.M), (
        "the surviving child's section title disappeared — its findings shipped "
        f"under the parent's question instead:\n{report}")
