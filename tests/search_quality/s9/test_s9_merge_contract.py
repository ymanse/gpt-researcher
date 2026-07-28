"""s9 (dedup harness): the roll-up must MERGE the tree, not concatenate it.

Measured defect (harness-search/DEDUP-HARNESS.md, benchmark round 4 + the 2026-07-28
verification run): every kept node answer is carried into the report >=70% verbatim
(lifted 12/12, max lift 100%), the report runs 120-133% of the kept answers, and it
ships 2 headings. `synthesize_node` joins a node's own answer to each child's summary
with "\n\n", so two siblings that researched near-identical questions state the same
finding twice.

OUTCOMES, NOT MECHANISM. Every assertion below is about what the shipped report must
contain. Nothing here says how the merge is found: cluster with an embedding, ask a
model, or match text — the tests do not care, and must not.

DETERMINISTIC MEANS MOCKED, NOT FORBIDDEN. The seams `tests/search_quality/s8` uses
are patched here for exactly that reason: `skill.embed_question` is replaced with an
offline stand-in and `tr.create_chat_completion` with a mock, so the implementation is
free to compare claims semantically or to consult a model while this file still runs
with no network, no embeddings service and no real LLM. The first s9 cut patched
NEITHER and asserted in its docstring that the assembly "makes no LLM call"; because
the measured redundancy is topical rather than lexical, that froze the implementation
out of the only approach that can reach the d1 thresholds and left deletion — which
`s2_aggregate_pct >= 80` correctly refuses — as the only move. See the intervention log
in DEDUP-HARNESS.md.

The mock LLM reply is generic on purpose. A live model's output is not deterministic
either, so nothing pinned here may depend on what it says: an implementation that
consults a model must still produce these outcomes when the answer is unhelpful.

What this file pins:

  (a) two node answers stating the SAME claim -> that claim appears ONCE
  (b) two node answers stating DIFFERENT claims -> both survive. Deleting content
      satisfies every redundancy metric, so "no silent deletion" is pinned as hard as
      the dedup itself (`s2_aggregate_pct >= 80` is the gate-level version).
  (c) a merged claim keeps the grounding both nodes earned. The frozen scorer matches
      `\\[(\\d{1,3})\\](?!\\()` and grounds a marker off the <=240 characters before it
      (harness-search/bench/score_report.py, score_s1), so this asks the grader's own
      question. Each source page here carries ONLY what that source says: the first cut
      appended every shared finding to every page, which grounded a migrated marker no
      matter whose wording survived — vacuous, and a property real pages never have.
      Node C's page does not state the growth claim at all, so moving [C] onto it fails.
  (d) the paste is gone, measured with scripts/rollup_scan.py's own 5-gram rule.
  (e) titled sections, not one undivided wall (the d1 gate's `headings_min >= 4`).

Three "LLM rewrites the merge, then re-derive [id] markers by fuzzy matching" designs
were each measured live to LOSE citations, one to citations_total=0 on a 33-node tree
(see the synthesize_node docstring). That is why (c) pins grounding as an outcome
rather than pinning any merge mechanism: whatever the implementation does, the markers
must still ground.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest import mock

import gpt_researcher.skills.tree_research as tr

ROOT_Q = "How do teams operate the transactional outbox pattern in production?"

URL_ROOT = "https://example.test/outbox-operations-overview"
URL_A = "https://example.test/payments-postmortem"
URL_B = "https://example.test/relay-incident-review"
URL_C = "https://example.test/cdc-migration-survey"

# --- the SHARED findings. Same figure, same nouns, different framing — the shape
# sibling nodes researching near-identical questions actually produce (the pending-node
# RCA), not a copy-paste. A word-overlap bar high enough to be safe never fires on these.
S1_A = ("The outbox table grew without bound in one payments cluster, which accumulated "
        "48,500,000 unpublished rows before any cleanup job existed.")
S1_B = ("Unpublished outbox rows accumulate without bound: that payments cluster reached "
        "48,500,000 unpublished rows before a cleanup job existed.")
S2_A = ("Relay throughput collapsed once the unpublished backlog passed 2,000,000 rows, "
        "because the polling query scanned the entire outbox table on every pass.")
S2_B = ("Once the backlog passed 2,000,000 rows the relay throughput collapsed, since "
        "every polling pass scanned the entire outbox table.")
S2_C = ("Polling scanned the entire outbox table on every pass, so relay throughput "
        "collapsed once the unpublished backlog passed 2,000,000 rows.")
S3_A = ("Consumers observed duplicate deliveries at a steady 0.4 percent of published "
        "events, so every downstream handler had to be written idempotent.")
S3_B = ("Roughly 0.4 percent of published events arrived twice, which forced every "
        "downstream consumer handler to be written idempotent.")

# --- findings only ONE node reports. None of these may be lost to the merge.
U_A1 = ("Dropping the poll interval to 250 milliseconds held end-to-end publish latency "
        "under 1.75 seconds at the ninety-fifth percentile.")
U_A2 = ("Backpressure from the relay stayed invisible until the publish lag metric was "
        "exported alongside the consumer lag dashboards.")
U_B1 = ("Two relay instances running without a leader lock double-published 12,400 events "
        "in a single afternoon.")
U_B2 = ("A partial outage left the relay reading a stale replica, so rows that were "
        "already published got republished throughout the failover.")
U_C1 = ("Change-data-capture connectors replaced the polling relay in 43% of the surveyed "
        "teams, removing the outbox table from the write path.")
U_C2 = ("Connector snapshots must be taken before the outbox table is dropped, otherwise "
        "the backfill window closes and cannot be recovered.")
U_C3 = ("Two surveyed teams kept a shadow outbox holding 6,200 rows for reconciliation "
        "after the connector cutover.")
U_C4 = ("Schema evolution on the source table forced a connector re-snapshot in every "
        "deployment that carried a wide event envelope.")

ROOT_ANSWER = ("Teams running an outbox in production keep reporting the same handful of "
               "recurring operational hazards, and the material gathered under this "
               "question describes each of them.")

ANSWER_A = " ".join([S1_A, S2_A, S3_A, U_A1, U_A2])
ANSWER_B = " ".join([S1_B, S2_B, S3_B, U_B1, U_B2])
ANSWER_C = " ".join([S2_C, U_C1, U_C2, U_C3, U_C4])

# Each page carries what ITS OWN source says and nothing another source says. That is
# what makes (c) a real question: a marker migrated onto a claim its page never made
# cannot ground, exactly as on a real page.
READ_DOCS = {URL_ROOT: ROOT_ANSWER, URL_A: ANSWER_A, URL_B: ANSWER_B, URL_C: ANSWER_C}

# citation ids are assigned over self.nodes INSERTION order (assemble_report), so:
ID_ROOT, ID_A, ID_B, ID_C = "1", "2", "3", "4"

# the frozen scorer's marker regex and grounding window, verbatim from
# harness-search/bench/score_report.py (CITE / score_s1). Mirrored, never imported: the
# scorer is frozen since s0 and gpt_researcher must not depend on the harness.
SCORER_CITE = re.compile(r"\[(\d{1,3})\](?!\()")
SCORER_WINDOW_CHARS = 240
SCORER_WINDOW_TOKENS = 20

# scripts/rollup_scan.py's lift measure, same constants (LIFT_PCT / MIN_NGRAMS)
LIFT_PCT = 70
MIN_NGRAMS = 50
_W = re.compile(r"[a-z0-9]+")
_BRACKETED = re.compile(r"\[[^\]]*\]")

# score_s6 / _claim_profile's context tokens, mirrored for the same reason
_STOP = {"that", "with", "this", "from", "have", "been", "were", "their", "which",
         "about", "into", "over", "only", "more", "than", "when", "after", "before",
         "while", "where", "also", "each", "other", "them", "they", "these", "those",
         "such", "some", "most", "many", "very", "will", "would", "could", "should",
         "there", "then", "what", "your", "does", "using", "used", "between"}


def _norm(text: str) -> str:
    """score_report.norm()."""
    return re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).strip()


def _ctx(text: str) -> set:
    """score_report.ctx_tokens()."""
    return {t for t in _norm(text).split()
            if len(t) >= 4 and not t.isdigit() and t not in _STOP}


def _toks(text: str) -> list:
    return [w for w in _W.findall(_BRACKETED.sub(" ", (text or "").lower())) if len(w) > 3]


def _ngrams(ws: list, n: int = 5) -> set:
    return {tuple(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def _traces(window: str, page: str) -> bool:
    """score_s1's match: any 3-token window of `window` carrying a 4+-character token
    that appears verbatim in `page`."""
    toks = window.split()
    page_n = _norm(page)
    for i in range(len(toks) - 2):
        win = toks[i:i + 3]
        if max(len(t) for t in win) >= 4 and " ".join(win) in page_n:
            return True
    return False


def _scorer_window(body: str, m: "re.Match") -> str:
    """Exactly what score_s1 reads before a marker, and no more: the last 20 normalized
    tokens of the preceding 240 characters. Markers normalize to bare digits here just
    as they do for the grader, so a run of [id]s pushes the claim out of view for both."""
    raw = body[max(0, m.start() - SCORER_WINDOW_CHARS):m.start()]
    return " ".join(_norm(raw).split()[-SCORER_WINDOW_TOKENS:])


def _window_grounds(span: str, page: str) -> bool:
    """score_s1's verdict on a marker placed at the end of `span`."""
    return _traces(" ".join(_norm(span).split()[-SCORER_WINDOW_TOKENS:]), page)


def _grounds_claim(body: str, cid: str, needle: str, citation_map: dict) -> bool:
    """The grader's own question about one citation: is there a marker for `cid` whose
    scorer window carries `needle` AND traces into the page `cid` points at? Both halves
    matter — a marker near the claim but off its own page is a fabricated grounding, and
    one on its own page but away from the claim grounds some other sentence."""
    needle_n = _norm(needle)
    page = READ_DOCS.get(citation_map.get(cid, ""), "")
    return any(needle_n in (w := _scorer_window(body, m)) and _traces(w, page)
               for m in SCORER_CITE.finditer(body) if m.group(1) == cid)


def _ungrounded_markers(body: str, citation_map: dict) -> list:
    """Every shipped marker the frozen scorer would fail to ground against its OWN
    source's page — i.e. an [id] sitting next to wording that source never supports."""
    bad = []
    for m in SCORER_CITE.finditer(body):
        cid = m.group(1)
        window = _scorer_window(body, m)
        if not _traces(window, READ_DOCS.get(citation_map.get(cid, ""), "")):
            bad.append((cid, window))
    return bad


# --- the offline seams -------------------------------------------------------------
# One axis per distinct finding in this fixture, keyed on that finding's own
# vocabulary. Two phrasings of one claim land on the same axis (cosine ~1.0); different
# claims land on different axes (cosine ~0.08), which is what a real embedding of these
# sentences does and what the topical redundancy makes impossible to see lexically. The
# trailing constant keeps every vector non-zero, so an implementation never has to
# special-case a text this stand-in does not recognise.
_CLAIM_AXES = (
    frozenset({"cleanup", "accumulate", "accumulated", "48"}),      # unbounded growth
    frozenset({"polling", "throughput", "collapsed", "scanned"}),   # relay collapse
    frozenset({"duplicate", "deliveries", "idempotent", "twice"}),  # duplicate delivery
    frozenset({"latency", "percentile", "milliseconds", "250"}),    # poll interval
    frozenset({"backpressure", "dashboards", "invisible"}),         # blind backpressure
    frozenset({"leader", "lock", "400"}),                           # double publish
    frozenset({"replica", "failover", "republished"}),              # failover replay
    frozenset({"connectors", "capture", "removing"}),               # cdc replacement
    frozenset({"snapshots", "backfill", "dropped"}),                # snapshot ordering
    frozenset({"shadow", "reconciliation", "cutover"}),             # shadow outbox
    frozenset({"schema", "evolution", "envelope"}),                 # schema evolution
    frozenset({"hazards", "recurring", "gathered"}),                # root framing
)

# Generic on purpose: three plain lines, the most common "section titles" shape. An
# implementation may parse it, ignore it, or never call the model at all — nothing
# pinned in this file may depend on which.
_LLM_REPLY = "Outbox growth and cleanup\nRelay reliability\nMigrating off polling\n"


async def _embed(text: str) -> list:
    """Offline stand-in for the embedding service (the seam s8 patches)."""
    words = set(_W.findall((text or "").lower()))
    return [float(len(axis & words)) for axis in _CLAIM_AXES] + [1.0]


def _skill() -> tr.TreeResearchSkill:
    """Same parent shape the s4/s8 tests use — TreeResearchSkill reads tone/websocket/
    headers/visited_urls off it in __init__."""
    parent = SimpleNamespace(
        query=ROOT_Q,
        cfg=SimpleNamespace(strategic_llm_provider="mock",
                            strategic_llm_model="mock", config_path=None),
        tone=None,
        websocket=None,
        headers={},
        visited_urls=set(),
    )
    return tr.TreeResearchSkill(parent)


def _node(nid: str, question: str, answer: str, url: str, depth: int,
          status=tr.NodeStatus.ANSWERED) -> tr.ResearchNode:
    n = tr.ResearchNode(id=nid, question=question, parent_id=None if depth == 0 else "0",
                        depth=depth)
    n.status = status
    n.answer_md = answer
    n.answer_digest = answer[:120]
    n.learnings = [answer]
    n.sources = [url]
    return n


def _tree() -> tr.TreeResearchSkill:
    """Root plus three siblings whose answers overlap on three findings and differ on
    eight — the redundancy the roll-up is supposed to collapse, at fixture scale."""
    skill = _skill()
    root = _node("0", ROOT_Q, ROOT_ANSWER, URL_ROOT, 0, tr.NodeStatus.EXPANDED)
    root.children = ["0.0", "0.1", "0.2"]
    skill.nodes = {
        "0": root,
        "0.0": _node("0.0", "How does the outbox table behave under load?", ANSWER_A, URL_A, 1),
        "0.1": _node("0.1", "What goes wrong with the outbox relay in production?", ANSWER_B, URL_B, 1),
        "0.2": _node("0.2", "How do teams migrate off outbox polling?", ANSWER_C, URL_C, 1),
    }
    skill._read_docs = dict(READ_DOCS)
    skill.embed_question = _embed
    return skill


async def _assemble():
    """Run the real assembly with both seams patched offline."""
    skill = _tree()
    with mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value=_LLM_REPLY)):
        result = await skill.assemble_report(ROOT_Q)
    return skill, result


def _body(report: str) -> str:
    """The report without its Citations block: "- [id] url" lines are data, and the
    scorer counts them as marker occurrences, so claim-level counting must not see
    them."""
    return report.split("\n## Citations", 1)[0]


def _sentence_with(body: str, needle: str) -> str:
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        if needle in sent:
            return sent
    return ""


def _sections(report: str) -> list:
    """(title, body) for every heading below the H1, Citations excluded."""
    parts = re.split(r"^(#{1,6})\s+(\S.*)$", report, flags=re.M)
    out = []
    for i in range(1, len(parts) - 2, 3):
        title, text = parts[i + 1].strip(), parts[i + 2]
        if parts[i] == "#" or title == "Citations":
            continue
        out.append((title, text))
    return out


async def test_the_same_claim_stated_by_two_nodes_is_reported_once():
    """(a) Three nodes report three shared findings between them. Each appears once."""
    _, result = await _assemble()
    body = _body(result["report_md"])

    for figure, claim, tellers in (("48,500,000", "unbounded outbox growth", 2),
                                   ("2,000,000", "relay collapse under backlog", 3),
                                   ("0.4 percent", "duplicate delivery rate", 2)):
        assert body.count(figure) == 1, (
            f"{claim} is stated by {tellers} nodes and must be merged into ONE "
            f"statement; '{figure}' occurs {body.count(figure)} times"
        )

    kept = _sentence_with(body, "48,500,000")
    missing = {"outbox", "unpublished", "rows", "payments", "cluster", "cleanup"} - _ctx(kept)
    assert not missing, (
        "the surviving statement must still BE the claim, not a stub of it — "
        f"missing {sorted(missing)} from: {kept!r}"
    )


async def test_claims_only_one_node_found_all_survive_the_merge():
    """(b) Deleting content satisfies every redundancy metric, so the merge is pinned
    against silent loss as hard as against duplication."""
    _, result = await _assemble()
    body = _body(result["report_md"])

    for figure in ("1.75", "12,400", "43%", "6,200"):
        assert body.count(figure) == 1, (
            f"a finding only one node reported was lost or duplicated: '{figure}' "
            f"occurs {body.count(figure)} times"
        )
    for token in ("backpressure", "failover", "backfill", "envelope"):
        assert token in body.lower(), (
            f"the merge dropped a finding no other node reported ({token!r}) — a shorter "
            "report bought by deleting content fails the frozen scorer's S2 floor"
        )
    # ... and the duplicated findings still collapse, in the same report
    assert body.count("2,000,000") == 1, (
        f"kept everything, merged nothing: '2,000,000' occurs {body.count('2,000,000')} "
        "times and three nodes state that one finding"
    )


async def test_merged_claim_keeps_every_grounding_the_nodes_earned():
    """(c) Two nodes, two sources, one claim: after the merge BOTH ids must still ground
    that claim by the frozen scorer's own rule, in the marker shape it reads."""
    # fixture guards — without these the question is not being asked
    assert _window_grounds(S1_A, READ_DOCS[URL_B]) and _window_grounds(S1_B, READ_DOCS[URL_A]), (
        "fixture: each phrasing of the shared claim must trace into the OTHER node's "
        "page, or keeping one wording could never keep both groundings and (c) would be "
        "unreachable rather than merely unmet"
    )
    assert not _window_grounds(S1_A, READ_DOCS[URL_C]) \
        and not _window_grounds(S1_B, READ_DOCS[URL_C]), (
        "fixture: node C's page must NOT support the growth claim — the first s9 cut put "
        "every shared finding on every page, so a migrated marker grounded no matter "
        "whose wording survived and this test proved nothing"
    )

    _, result = await _assemble()
    body = _body(result["report_md"])
    citation_map = result["citation_map"]

    assert body.count("48,500,000") == 1, (
        "precondition for this test: the shared claim must be ONE statement before "
        "asking whether that statement carries both groundings"
    )
    for cid, url in ((ID_A, URL_A), (ID_B, URL_B)):
        assert _grounds_claim(body, cid, "48,500,000", citation_map), (
            f"[{cid}] ({url}) grounded this claim before the merge and must still ground "
            "it after: a marker whose scorer window carries the claim and traces into "
            "that source's own page. A merge that re-derives markers by fuzzy matching "
            "is what drove citations_total to 0 live"
        )

    ungrounded = _ungrounded_markers(body, citation_map)
    assert not ungrounded, (
        "a marker was moved onto wording its own source never supports; the frozen "
        f"scorer grounds none of these: {ungrounded}"
    )
    assert not re.search(r"\[\d{1,3}\]\(", body), (
        "a marker rendered as a markdown link is invisible to the frozen scorer's "
        r"`\[(\d{1,3})\](?!\()` regex"
    )

    body_ids = {m.group(1) for m in SCORER_CITE.finditer(body)}
    assert body_ids == {ID_ROOT, ID_A, ID_B, ID_C}, (
        "every node earned a grounded citation before the merge; dropping one is how a "
        f"merge cheats the grounding check instead of preserving it: {sorted(body_ids)}"
    )
    listed = {m.group(1) for m in SCORER_CITE.finditer(result["report_md"][len(body):])}
    assert body_ids == listed, (
        f"the Citations block must list exactly the ids the body uses: "
        f"body={sorted(body_ids)} listed={sorted(listed)}"
    )


async def test_node_answers_are_not_carried_into_the_report_verbatim():
    """(d) The lifted_nodes measure, scripts/rollup_scan.py's rule, at fixture scale."""
    skill, result = await _assemble()
    rep_ng = _ngrams(_toks(result["report_md"]))

    lifted, scanned = [], 0
    for nid, node in skill.nodes.items():
        g = _ngrams(_toks(node.answer_md))
        if len(g) < MIN_NGRAMS:
            continue
        scanned += 1
        pct = round(100 * len(g & rep_ng) / len(g))
        if pct >= LIFT_PCT:
            lifted.append((nid, pct))

    assert scanned >= 3, (
        f"fixture guard: {scanned} node answers reached {MIN_NGRAMS} 5-grams — a scan "
        "that measures nothing must never read as 'no redundancy'"
    )
    assert len(lifted) <= 1, (
        f"{len(lifted)} node answers are still >={LIFT_PCT}% present in the report "
        f"({lifted}) — that is concatenation. The d1 gate allows exactly one "
        "(lifted_nodes_max <= 1): the node whose wording a shared claim keeps, which is "
        "what preserves that node's citation grounding"
    )


async def test_report_is_divided_into_titled_sections():
    """(e) The d1 gate reads headings_min >= 4 with this regex; the measured report
    ships 2 (the query H1 and the Citations block)."""
    _, result = await _assemble()
    report = result["report_md"]
    headings = re.findall(r"^#{1,6}\s+(\S.*)$", report, re.M)

    assert len(headings) >= 4, (
        f"the report is one undivided wall — {len(headings)} headings, the d1 gate "
        f"needs >= 4: {headings}"
    )
    sections = _sections(report)
    assert len(sections) >= 2, (
        f"the query title and the Citations block are not sections: {headings}"
    )
    for title, text in sections:
        assert SCORER_CITE.search(text), (
            f"section {title!r} carries no cited finding — every finding in this fixture "
            "is grounded, so a section with no marker is a divider inflating the heading "
            "count, not a theme the report was organised into"
        )
