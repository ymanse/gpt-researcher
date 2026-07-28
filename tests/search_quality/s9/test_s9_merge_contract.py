"""s9 (dedup harness): the roll-up must MERGE the tree, not concatenate it.

Measured defect (DEDUP-HARNESS.md, benchmark round 4 + the 2026-07-28 verification
run): every kept node answer is carried into the report >=70% verbatim (lifted 12/12,
max lift 100%), the report runs 120-133% of the kept answers, and it ships 2 headings.
`synthesize_node` joins a node's own answer to each child's summary with "\n\n" — so
two siblings that researched near-identical questions state the same finding twice.

What this file pins, and why each shape is the way it is:

  (a) two node answers stating the SAME claim -> that claim appears ONCE
  (b) two node answers stating DIFFERENT claims -> both survive. Deleting content
      satisfies every redundancy metric, so "no silent deletion" is pinned as
      hard as the dedup itself (the frozen scorer's S2 >= 80 is the gate-level
      version of this same anti-cheat).
  (c) when two nodes' groundings collapse into one statement, BOTH [id] markers
      stay next to it, in the shape the FROZEN scorer can read: score_s1 matches
      `\\[(\\d{1,3})\\](?!\\()` and grounds a marker off the <=240 characters
      before it (bench/score_report.py), so this test asks the same question the
      grader will — is there a marker for that id whose scorer window carries the
      claim. Three "LLM rewrites the merge, then re-derive [id] by fuzzy matching"
      designs were each measured live to LOSE citations, one to citations_total=0
      on a 33-node tree (see the synthesize_node docstring), which is why the
      contract is pinned on the OUTCOME (claim stated once, both groundings kept)
      and not on any particular merge mechanism.
  (d) the paste is gone: measured with scripts/rollup_scan.py's own 5-gram rule
      (LIFT_PCT 70, MIN_NGRAMS 50), at most ONE node answer may still be >=70%
      present. One is what the d1 gate allows (`lifted_nodes_max <= 1`) and it is
      what a first-wins merge legitimately produces: whichever node's wording is
      chosen as canonical for a shared claim keeps its own text verbatim, which is
      exactly what keeps that node's citations grounded. Demanding zero here would
      be stricter than the instrument that decides the build.
  (e) titled sections, not one undivided wall (the d1 gate's `headings_min >= 4`).

Deterministic: `assemble_report` performs no retrieval and makes no LLM call
(create_chat_completion appears twice in tree_research.py and both sites are
upstream of it), so these tests need no network, no embeddings service and no
model — the fixture supplies the node answers and the scraped documents directly.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import gpt_researcher.skills.tree_research as tr

ROOT_Q = "How do teams operate the transactional outbox pattern in production?"

URL_ROOT = "https://example.test/outbox-overview"
URL_A = "https://example.test/payments-postmortem"
URL_B = "https://example.test/relay-incident-review"
URL_C = "https://example.test/cdc-migration-survey"

# --- the three SHARED findings, each stated by more than one node in its own wording.
# Same figure, same nouns, different framing — the shape sibling nodes researching
# near-identical questions actually produce (pending_rca.md), not a copy-paste.
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

# --- findings only ONE node has. None of these may be lost to the merge.
U_A = ("Dropping the poll interval to 250 ms held end-to-end publish latency under "
       "1.75 seconds at the ninety-fifth percentile.")
U_A2 = ("Backpressure from the relay stayed invisible until the publish lag metric was "
        "exported alongside the consumer lag dashboards.")
U_B = ("Two relay instances running without a leader lock double-published 12,400 events "
       "in a single afternoon.")
U_B2 = ("A partial outage left the relay reading a stale replica, so rows already "
        "published were republished during failover.")
U_C = ("Change-data-capture connectors replaced the polling relay in 43% of the surveyed "
       "teams, removing the outbox table from the write path.")
U_C2 = ("Connector snapshots must be taken before the outbox table is dropped, otherwise "
        "the backfill window closes and cannot be recovered.")

ROOT_ANSWER = ("Production outbox deployments share three operational failure modes: "
               "unbounded table growth, relay throughput collapse under backlog, and "
               "duplicate delivery that consumers must absorb.")

ANSWER_A = " ".join([S1_A, S2_A, S3_A, U_A, U_A2])
ANSWER_B = " ".join([S1_B, S2_B, S3_B, U_B, U_B2])
ANSWER_C = " ".join([S2_C, U_C, U_C2, S1_B, S3_A])

# Every shared finding is in EVERY source page: the siblings are reporting the same
# incidents from overlapping coverage. That is what makes (c) a fair ask — a merged
# statement can only keep both markers if both pages really do support the claim.
_SHARED_COVERAGE = "\n".join([S1_A, S1_B, S2_A, S2_B, S2_C, S3_A, S3_B])
READ_DOCS = {
    URL_ROOT: ROOT_ANSWER + "\n" + _SHARED_COVERAGE,
    URL_A: ANSWER_A + "\n" + _SHARED_COVERAGE,
    URL_B: ANSWER_B + "\n" + _SHARED_COVERAGE,
    URL_C: ANSWER_C + "\n" + _SHARED_COVERAGE,
}

# citation ids are assigned over self.nodes INSERTION order (assemble_report), so:
ID_ROOT, ID_A, ID_B, ID_C = "1", "2", "3", "4"

# the frozen scorer's marker regex and grounding window, verbatim from
# bench/score_report.py (CITE / score_s1). Mirrored, never imported: the scorer is
# frozen since s0 and gpt_researcher must not depend on the harness.
SCORER_CITE = re.compile(r"\[(\d{1,3})\](?!\()")
SCORER_WINDOW = 240

# scripts/rollup_scan.py's lift measure, same constants (LIFT_PCT / MIN_NGRAMS)
LIFT_PCT = 70
MIN_NGRAMS = 50
_W = re.compile(r"[a-z0-9]+")
_BRACKETED = re.compile(r"\[[^\]]*\]")


def _toks(text: str) -> list[str]:
    return [w for w in _W.findall(_BRACKETED.sub(" ", text.lower())) if len(w) > 3]


def _ngrams(ws: list[str], n: int = 5) -> set:
    return {tuple(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


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
    six — the redundancy the roll-up is supposed to collapse, at fixture scale."""
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
    return skill


def _body(report: str) -> str:
    """The report without its Citations block: "- [id] url" lines are data, and the
    scorer counts them as marker occurrences, so claim-level counting must not see them."""
    return report.split("\n## Citations", 1)[0]


def _sentence_with(body: str, needle: str) -> str:
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        if needle in sent:
            return sent
    return ""


def _marker_grounds(body: str, cid: str, needle: str) -> bool:
    """The grader's own question: is there a marker for `cid` whose 240-character
    scorer window carries `needle`? (bench/score_report.py score_s1)"""
    return any(needle in body[max(0, m.start() - SCORER_WINDOW):m.start()]
               for m in SCORER_CITE.finditer(body) if m.group(1) == cid)


async def test_the_same_claim_stated_by_two_nodes_is_reported_once():
    """(a) Three nodes report the same three findings. Each must appear once."""
    report = (await _tree().assemble_report(ROOT_Q))["report_md"]
    body = _body(report)

    for figure, claim in (("48,500,000", "unbounded outbox growth"),
                          ("2,000,000", "relay collapse under backlog"),
                          ("0.4 percent", "duplicate delivery rate")):
        assert body.count(figure) == 1, (
            f"{claim} is stated by more than one node and must be merged into ONE "
            f"statement; '{figure}' occurs {body.count(figure)} times"
        )

    kept = _sentence_with(body, "48,500,000")
    _, ctx = tr._claim_profile(kept)
    missing = {"outbox", "unpublished", "rows", "payments", "cluster", "cleanup"} - ctx
    assert not missing, (
        "the surviving statement must still BE the claim, not a stub of it — "
        f"missing {sorted(missing)} from: {kept!r}"
    )


async def test_claims_only_one_node_found_all_survive_the_merge():
    """(b) Deleting content satisfies every redundancy metric, so the merge is pinned
    against silent loss as hard as against duplication."""
    report = (await _tree().assemble_report(ROOT_Q))["report_md"]
    body = _body(report)

    for figure in ("1.75", "12,400", "43%"):
        assert body.count(figure) == 1, (
            f"a finding only one node reported was lost or duplicated: '{figure}' "
            f"occurs {body.count(figure)} times"
        )
    for token in ("backpressure", "failover", "backfill"):
        assert token in body.lower(), (
            f"the merge dropped a finding no other node reported ({token!r}) — a shorter "
            "report bought by deleting content fails the frozen scorer's S2 floor"
        )
    # ... and the duplicated findings still collapse, in the same report
    assert body.count("2,000,000") == 1, "kept everything, merged nothing"


async def test_merged_claim_keeps_every_grounding_the_nodes_earned():
    """(c) Two nodes, two sources, one claim: after the merge BOTH ids must still
    ground that claim by the frozen scorer's own rule, in the marker shape it reads."""
    report = (await _tree().assemble_report(ROOT_Q))["report_md"]
    body = _body(report)

    assert body.count("48,500,000") == 1, (
        "precondition for this test: the shared claim must be one statement before "
        "asking whether that statement carries both groundings"
    )
    for cid, url in ((ID_A, URL_A), (ID_B, URL_B)):
        assert _marker_grounds(body, cid, "48,500,000"), (
            f"[{cid}] ({url}) supported this claim before the merge and must still sit "
            "within the scorer's 240-character window of it — a merge that re-derives "
            "markers by fuzzy matching is what drove citations_total to 0 live"
        )
    assert not re.search(r"\[\d{1,3}\]\(", body), (
        "a marker rendered as a markdown link is invisible to the frozen scorer's "
        r"`\[(\d{1,3})\](?!\()` regex"
    )
    body_ids = {m.group(1) for m in SCORER_CITE.finditer(body)}
    listed = {m.group(1) for m in SCORER_CITE.finditer(report[len(body):])}
    assert body_ids and body_ids == listed, (
        f"the Citations block must list exactly the ids the body uses: body={sorted(body_ids)} "
        f"listed={sorted(listed)}"
    )


async def test_node_answers_are_not_carried_into_the_report_verbatim():
    """(d) The lifted_nodes measure, scripts/rollup_scan.py's rule, at fixture scale."""
    skill = _tree()
    report = (await skill.assemble_report(ROOT_Q))["report_md"]
    rep_ng = _ngrams(_toks(report))

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
        "(lifted_nodes_max <= 1): the node whose wording a shared claim keeps, which "
        "is what preserves that node's citation grounding"
    )


async def test_report_is_divided_into_titled_sections():
    """(e) The d1 gate reads headings_min >= 4 with this regex; the measured report
    ships 2 (the query H1 and the Citations block)."""
    report = (await _tree().assemble_report(ROOT_Q))["report_md"]
    headings = re.findall(r"^#{1,6}\s+(\S.*)$", report, re.M)

    assert len(headings) >= 4, (
        f"the report is one undivided wall — {len(headings)} headings, the d1 gate "
        f"needs >= 4: {headings}"
    )
    titled = [h for h in headings if h.strip() not in (ROOT_Q, "Citations")]
    assert len(titled) >= 2, (
        f"the query title and the Citations block are not sections: {headings}"
    )
