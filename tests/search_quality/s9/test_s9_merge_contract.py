"""s9 (dedup harness): the roll-up must MERGE the tree, not concatenate it — and
"merge" means one statement per CLAIM, never one statement per topic.

Measured defect (harness-search/DEDUP-HARNESS.md, benchmark round 4 + the 2026-07-28
verification run): every kept node answer is carried into the report >=70% verbatim
(lifted 12/12, max lift 100%), the report runs 120-133% of the kept answers, and it
ships 2 headings. `synthesize_node` joins a node's own answer to each child's summary
with "\\n\\n", so two nodes that researched near-identical questions state the same
finding twice.

OUTCOMES, NOT MECHANISM. Every assertion below is about what the shipped report must
contain. Nothing here says how the merge is found: cluster with an embedding, ask a
model, or match text — the tests do not care, and must not.

DETERMINISTIC MEANS MOCKED, NOT FORBIDDEN. The seams `tests/search_quality/s8` uses
are patched here for exactly that reason: `skill.embed_question` is replaced with an
offline stand-in and `tr.create_chat_completion` with one too, so the implementation is
free to compare claims semantically or to consult a model while this file still runs
with no network, no embeddings service and no real LLM.

WHY THE FIXTURE IS THE REAL CORPUS. Four implementations (word overlap, LSA cosine,
coverage/vocabulary exhaustion, item-budget selection) reached the ratio and heading
thresholds and then lost facts, because all four decided on how much two sentences have
IN COMMON and none could see what only one of them said. The last one dropped 2 of
`denorm-derived-table`'s 8 golden facts (S2 63 -> 38, `s2_min_delta` -25) at exactly two
collapses. Both collapses are pinned below, read VERBATIM out of the frozen corpus
(`harness-search/no_read/dedup/corpus/denorm-derived-table.tree.json`, read-only) so
neither can be softened into something a word-overlap rule happens to pass:

  * node 0.0.2.0 (Oracle docs) — "a complete refresh ... likewise fully recomputes the
    view" was merged away in favour of "an incremental refresh eliminates the need to
    rebuild materialized views from scratch". Same subject, same vocabulary, OPPOSITE
    mechanisms; golden fact 2 went with the deleted sentence.
  * node 0.3.1 (MySQL docs) — the closure-table DEFINITION ("a separate table stores all
    ancestor-descendant relationships") was merged away in favour of the closure-table
    read-vs-write TRADE-OFF. Definition and trade-off are different claims about one
    thing; golden fact 8 went with the deleted sentence.

Formally (arXiv:2509.08304, Answerable Question Sets): two texts are EQUIVALENT only
when neither answers a question the other cannot. When both sides have unique answerable
questions the relation is OVERLAP, and overlap must not be collapsed. Similarity scores
the intersection; the decision lives in the difference.

NO SIMILARITY THRESHOLD CAN PASS THIS FIXTURE, and that is measured, not asserted by
fiat: `_assert_similarity_cannot_decide` recomputes, over these verbatim sentences and
with the very embedding this file patches in, that some pair which MUST merge scores
BELOW a pair which must NOT. Any threshold low enough to merge the first also merges the
second. That inversion is not a fixture trick — it is the harness's central measurement:
sibling redundancy here is TOPICAL, not lexical (repeated 5-grams 0-2%, max section-pair
Jaccard 0.22), while the same-topic-different-claim pairs share a whole vocabulary.

What this file pins:

  (a) two node answers stating the SAME claim -> that claim appears ONCE
  (b) two node answers stating DIFFERENT claims -> both survive. Deleting content
      satisfies every redundancy metric, so "no silent deletion" is pinned as hard as
      the dedup itself (`s2_aggregate_pct >= 80` / `s2_min_delta >= -5` are the
      gate-level version).
  (c) a merged claim keeps the grounding the nodes earned. The frozen scorer matches
      `\\[(\\d{1,3})\\](?!\\()` and grounds a marker off the <=240 characters before it
      (harness-search/bench/score_report.py, score_s1), so this asks the grader's own
      question. Each source page here carries ONLY what that source says: an earlier cut
      appended every shared finding to every page, which grounded a migrated marker no
      matter whose wording survived — vacuous, and a property real pages never have.
  (d) the paste is gone, measured with scripts/rollup_scan.py's own 5-gram rule.
  (e) titled sections, not one undivided wall (the d1 gate's `headings_min >= 4`).
  (f) the two real collapses above: same topic, different claim, BOTH survive.

Three "LLM rewrites the merge, then re-derive [id] markers by fuzzy matching" designs
were each measured live to LOSE citations, one to citations_total=0 on a 33-node tree
(see the synthesize_node docstring). That is why (c) pins grounding as an outcome
rather than pinning any merge mechanism: whatever the implementation does, the markers
must still ground.

THE OFFLINE MODEL STAND-IN, and what it is willing to answer. Because the inversion
above means no similarity score can make this call, the decision has to come from
reading the sentences — which in production is a model call, and here is
`_model`. It answers ANY prompt that quotes at least two of the fixture's sentences by
enumerating, per quoted sentence, what that sentence says which no other quoted sentence
says (the operator DEDUP-HARNESS.md prescribes: the LLM ENUMERATES each side's unique
content rather than classifying a relation). One line per quoted sentence:

    "<a verbatim fragment of that sentence>" => UNIQUE: none
    "<a verbatim fragment of that sentence>" => UNIQUE: <what only this one says>

`UNIQUE: none` means some other quoted sentence answers everything this one answers, so
the pair/cluster may collapse to one member. Anything else means it may not. The
implementation chooses its own prompt, its own batching (pairwise or whole-cluster) and
its own parsing; the only thing it must do to get an answer is quote the sentences it is
asking about. A prompt quoting fewer than two known sentences gets a generic three-line
outline instead — so an outline/section-title call still works, and a comparison call
that fails to quote its inputs gets nothing usable and must fail closed (not merge),
which is the safe direction: not merging costs `synthesis_ratio_pct_max`, wrongly
merging costs facts, and `s2_min_delta >= -5` is the tighter bound.
"""
from __future__ import annotations

import re
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import gpt_researcher.skills.tree_research as tr

ROOT_Q = "How should a denormalized derived table be maintained: incremental refresh, or a full rebuild?"

URL_ROOT = "https://example.test/derived-table-maintenance-overview"
URL_ORACLE = "https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/refreshing-materialized-views.html"
URL_PGIVM = "https://github.com/sraoss/pg_ivm"
URL_MYSQL = "https://dev.mysql.com/doc/refman/8.0/en/with.html"

# ---------------------------------------------------------------------------------
# VERBATIM out of no_read/dedup/corpus/denorm-derived-table.tree.json (nodes[].answer).
# Only a leading markdown list bullet is dropped; no word is changed. Node ids are the
# corpus's own, so a reader can go back and check every sentence.
# ---------------------------------------------------------------------------------

# --- node 0.0.2.0 (Oracle docs) -------------------------------------------------
O_FAST = (
    "Oracle's official documentation identifies **Fast Refresh using materialized view "
    "logs** as its vendor-authoritative, delta-based incremental maintenance mechanism "
    "for materialized views, including those with aggregate columns — explicitly "
    "positioned as an alternative to full recomputation."
)
# the sentence that SURVIVED the collapse
O_INCR = (
    "**The core mechanism:** According to Oracle's Data Warehousing Guide (19c and 23c "
    'editions), "An incremental refresh eliminates the need to rebuild materialized '
    'views from scratch. Thus, processing only the changes can result in a very fast '
    'refresh time."'
)
# the sentence that was DELETED in its place, and took golden fact 2 with it
O_FULL = (
    'Without a materialized view log, Oracle states plainly, "the database must '
    "reexecute the materialized view query to refresh the materialized view. This "
    'process is called a complete refresh" — the functional analog to PostgreSQL\'s '
    "`REFRESH MATERIALIZED VIEW`, which likewise fully recomputes the view."
)

# --- node 0.0.2.1 (pg_ivm docs) — the same incremental claim, framed for another tool
P_IVM = (
    "The README's opening description frames this directly: *\"Incremental View "
    "Maintenance (IVM) is a way to make materialized views up-to-date in which only "
    "incremental changes are computed and applied on views rather than recomputing the "
    'contents from scratch as REFRESH MATERIALIZED VIEW does."*'
)
P_TRIG = (
    "Mechanistically, per the documentation, this immediate maintenance is implemented "
    "via **AFTER triggers** automatically installed on the base tables referenced by "
    "the view."
)
P_IMM = (
    'The documentation then states plainly: *"pg_ivm provides a kind of immediate '
    "maintenance, in which materialized views are updated immediately in AFTER triggers "
    'when a base table is modified."*'
)

# --- node 0.3.1 (MySQL docs) ----------------------------------------------------
# the DEFINITION, deleted by the collapse; it carries golden fact 8
M_DEF = (
    "w3tutorials.net's guide is more explicit about the historical limitation and "
    "alternatives: it lists path enumeration and closure tables as alternative "
    'hierarchical models, describing closure tables as a "separate table stores all '
    'ancestor-descendant relationships," offering "fast queries" and depth tracking but '
    "requiring maintenance of the closure table on inserts/updates."
)
# the TRADE-OFF, stated three times by three different sources in one node answer
M_TRADE1 = (
    "It ultimately **recommends the adjacency list model combined with recursive CTEs** "
    "as the primary modern approach, but adds a performance caveat: for very deep or "
    'large hierarchies (100+ levels), it suggests "precompute descendants with a '
    'closure table" as an alternative to recursive CTEs, essentially saying closure '
    "tables win for read-heavy, deep-hierarchy workloads, while adjacency list + "
    "recursive CTE is simplest for insert/update-heavy or shallow trees."
)
M_TRADE2 = (
    "Stack Overflow's community-wiki answer on hierarchical data storage makes a "
    'similar general trade-off point (not MySQL-specific): "fast read times (nested '
    'set/closure-table-like models) or fast write times (adjacency list)... usually you '
    'end up with a combination."'
)
M_TRADE3 = (
    "**Summary of the disagreement/nuance:** There's no real factual clash among these "
    "third-party sources — they agree adjacency list is simpler to write/insert, "
    "closure tables (or nested sets) are faster to read especially for deep/large "
    "hierarchies, and that recursive CTEs (available in MySQL only from 8.0 onward) "
    "made adjacency-list traversal practical without giving up its simplicity."
)

ROOT_ANSWER = (
    "The material gathered under this question comes from three vendor documentation "
    "sets, and each of them answers a different part of it."
)

ANSWER_ORACLE = "\n\n".join([O_FAST, O_INCR, O_FULL])
ANSWER_PGIVM = "\n\n".join([P_IMM, P_TRIG, P_IVM])
ANSWER_MYSQL = "\n\n".join([M_DEF, M_TRADE1, M_TRADE2, M_TRADE3])

# Each page carries what ITS OWN source says and nothing another source says. That is
# what makes (c) a real question: a marker migrated onto a claim its page never made
# cannot ground, exactly as on a real page.
READ_DOCS = {URL_ROOT: ROOT_ANSWER, URL_ORACLE: ANSWER_ORACLE,
             URL_PGIVM: ANSWER_PGIVM, URL_MYSQL: ANSWER_MYSQL}

# citation ids are assigned over self.nodes INSERTION order (assemble_report), so:
ID_ROOT, ID_ORACLE, ID_PGIVM, ID_MYSQL = "1", "2", "3", "4"

# --- what each sentence CLAIMS -----------------------------------------------------
# Sentences sharing a claim id are equivalent (neither answers a question the other
# cannot) and must collapse to one. Sentences with different claim ids must all survive,
# INCLUDING the two same-topic pairs the four failed implementations collapsed.
CLAIMS = {
    "incremental-applies-deltas-not-a-full-rebuild": (O_INCR, P_IVM),
    "complete-refresh-recomputes-the-whole-view": (O_FULL,),
    "fast-refresh-is-oracle's-named-delta-mechanism": (O_FAST,),
    "pg_ivm-maintains-views-inside-after-triggers": (P_TRIG,),
    "pg_ivm-maintenance-is-immediate-not-deferred": (P_IMM,),
    "a-closure-table-stores-every-ancestor-descendant-pair": (M_DEF,),
    "closure-tables-read-faster-adjacency-lists-write-faster":
        (M_TRADE1, M_TRADE2, M_TRADE3),
    "the-question's-material-comes-from-three-vendors": (ROOT_ANSWER,),
}

# A short, marker-free, verbatim fragment that identifies each sentence in the shipped
# report. Presence of the fragment IS presence of the claim: each one is the wording the
# claim lives in, so a stub or a re-worded husk does not match it.
KEY = {
    O_FAST: "delta-based incremental maintenance mechanism",
    O_INCR: "eliminates the need to rebuild materialized views from scratch",
    O_FULL: "likewise fully recomputes the view",
    P_IVM: "only incremental changes are computed and applied on views",
    P_TRIG: "automatically installed on the base tables",
    P_IMM: "provides a kind of immediate maintenance",
    M_DEF: "separate table stores all ancestor-descendant relationships",
    M_TRADE1: "closure tables win for read-heavy, deep-hierarchy workloads",
    M_TRADE2: "fast read times (nested set/closure-table-like models)",
    M_TRADE3: "faster to read especially for deep/large hierarchies",
    ROOT_ANSWER: "comes from three vendor documentation sets",
}
CLAIM_OF = {s: cid for cid, sents in CLAIMS.items() for s in sents}
ALL_SENTENCES = list(CLAIM_OF)

# --- frozen instruments, mirrored (never imported) ---------------------------------
# The scorer is frozen since s0 and gpt_researcher must not depend on the harness, so
# its regexes are reproduced here. Keep in sync if bench/ is ever re-frozen.
SCORER_CITE = re.compile(r"\[(\d{1,3})\](?!\()")          # score_report.CITE
SCORER_WINDOW_CHARS = 240                                  # score_s1
SCORER_WINDOW_TOKENS = 20                                  # score_s1

# bench/golden/denorm-derived-table.json, facts[1] and facts[7] — the two this corpus
# lost when the pairs above were collapsed. They are the reason (f) is a contract and
# not a preference.
GOLDEN_COMPLETE_REFRESH = re.compile(
    r"(?i)refresh\s+materialized\s+view[^.\n]{0,120}"
    r"(?:completely\s+replac\w+|discard(?:s|ed)|full(?:y)?\s+re(?:comput|build)\w*"
    r"|not\s+incremental|recomputes?\s+the\s+entire)")
GOLDEN_CLOSURE_TABLE = re.compile(
    r"(?i)closure[- ]table[^.\n]{0,200}"
    r"(?:every\s+(?:path|ancestor|pair)|all\s+ancestor\w*"
    r"|ancestor[–— -]descendant\s+pairs?|self[- ]referenc\w+|depth\s+column"
    r"|more\s+(?:storage|space|rows)|trades?\s+(?:space|storage))")

# scripts/rollup_scan.py's lift measure, same constants (LIFT_PCT / MIN_NGRAMS)
LIFT_PCT = 70
MIN_NGRAMS = 50
_W = re.compile(r"[a-z0-9]+")
_BRACKETED = re.compile(r"\[[^\]]*\]")


def _norm(text: str) -> str:
    """score_report.norm()."""
    return re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).strip()


def _flat(text: str) -> str:
    """The text as a reader sees the claims in it: citation markers removed and runs of
    whitespace collapsed, so a marker inserted mid-sentence cannot hide a fragment."""
    return re.sub(r"\s+", " ", _BRACKETED.sub(" ", text or "")).strip()


def _content(text: str) -> list:
    """rollup_scan.toks(): the 4+-character words, markers stripped."""
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
    tokens of the preceding 240 characters."""
    raw = body[max(0, m.start() - SCORER_WINDOW_CHARS):m.start()]
    return " ".join(_norm(raw).split()[-SCORER_WINDOW_TOKENS:])


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
# A surface embedding, deliberately: one axis per content word this fixture uses, plus a
# constant tail so no text embeds to the zero vector. It is NOT a claim oracle — an
# idealised "same claim -> same vector" stand-in would hand the decision back to a cosine
# threshold and let the fifth similarity implementation pass a test written because the
# first four failed. What this returns is what a real encoder can see: shared words.
_AXES = sorted({t for s in ALL_SENTENCES for t in _content(s)})


async def _embed(text: str) -> list:
    """Offline stand-in for the embedding service (the seam s8 patches)."""
    counts = Counter(_content(text))
    return [float(counts[a]) for a in _AXES] + [0.1]


def _cosine(a: list, b: list) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_content(a)), set(_content(b))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


# Generic on purpose: three plain lines, the most common "section titles" shape. An
# implementation may parse it, ignore it, or never ask for it — nothing pinned in this
# file may depend on which.
_OUTLINE_REPLY = ("Keeping a derived table up to date\n"
                  "Rebuilding it from scratch\n"
                  "Storing the hierarchy\n")


async def _model(*args, **kwargs) -> str:
    """Offline stand-in for the strategic model (the seam s8 patches).

    Enumerates, for each fixture sentence quoted in the prompt, what that sentence says
    which no other quoted sentence says — `UNIQUE: none` when another quoted sentence
    covers it entirely. See the module docstring for the full contract. Sentences it
    does not recognise are simply not listed, so an implementation that cannot find its
    input in the reply has to fail closed.
    """
    parts = [str(m.get("content", "")) for m in kwargs.get("messages", [])
             if isinstance(m, dict)]
    prompt = _flat(" ".join(parts) if parts else " ".join(str(a) for a in args))
    quoted = [s for s in ALL_SENTENCES if KEY[s] in prompt]
    if len(quoted) < 2:
        return _OUTLINE_REPLY
    seen = Counter(CLAIM_OF[s] for s in quoted)
    return "\n".join(
        f'"{KEY[s]}" => UNIQUE: '
        + ("none" if seen[CLAIM_OF[s]] > 1 else CLAIM_OF[s].replace("-", " "))
        for s in quoted) + "\n"


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
    """The corpus's own shape at fixture scale: a root and the three answered nodes
    whose sentences the four failed implementations merged wrongly."""
    skill = _skill()
    root = _node("0", ROOT_Q, ROOT_ANSWER, URL_ROOT, 0, tr.NodeStatus.EXPANDED)
    root.children = ["0.0.2.0", "0.0.2.1", "0.3.1"]
    skill.nodes = {
        "0": root,
        "0.0.2.0": _node("0.0.2.0", "What does Oracle's own documentation say about "
                         "materialized view logs and FAST refresh?",
                         ANSWER_ORACLE, URL_ORACLE, 1),
        "0.0.2.1": _node("0.0.2.1", "What does the pg_ivm project's own documentation "
                         "specify for incremental view maintenance?",
                         ANSWER_PGIVM, URL_PGIVM, 1),
        "0.3.1": _node("0.3.1", "What does MySQL's own reference documentation "
                       "recommend for storing hierarchies?",
                       ANSWER_MYSQL, URL_MYSQL, 1),
    }
    skill._read_docs = dict(READ_DOCS)
    skill.embed_question = _embed
    return skill


async def _assemble():
    """Run the real assembly with both seams patched offline."""
    skill = _tree()
    with mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(side_effect=_model)):
        result = await skill.assemble_report(ROOT_Q)
    return skill, result


def _body(report: str) -> str:
    """The report without its Citations block: "- [id] url" lines are data, and the
    scorer counts them as marker occurrences, so claim-level counting must not see
    them."""
    return report.split("\n## Citations", 1)[0]


def _present(body: str, *sentences: str) -> list:
    """Which of these sentences the shipped report still states."""
    flat = _flat(body)
    return [s for s in sentences if KEY[s] in flat]


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


async def _assert_similarity_cannot_decide() -> None:
    """The measurement that makes (f) more than an opinion.

    Over these verbatim sentences, and with the embedding this file actually patches in,
    the most similar pairs are NOT the ones that must merge. So for each measure there is
    a threshold-free contradiction: any cut-off low enough to merge every equivalent pair
    also merges a pair that states two different claims. Similarity may SELECT candidates;
    it cannot DECIDE. If this ever stops holding, the fixture has been softened and the
    contract below is no longer the one four implementations failed.
    """
    must_merge = [(a, b) for sents in CLAIMS.values() if len(sents) > 1
                  for i, a in enumerate(sents) for b in sents[i + 1:]]
    must_keep = [(a, b) for a in ALL_SENTENCES for b in ALL_SENTENCES
                 if a < b and CLAIM_OF[a] != CLAIM_OF[b]]
    vec = {s: await _embed(s) for s in ALL_SENTENCES}

    for name, score in (("word overlap", lambda a, b: _jaccard(a, b)),
                        ("embedding cosine", lambda a, b: _cosine(vec[a], vec[b]))):
        weakest_merge = min(score(a, b) for a, b in must_merge)
        strongest_keep = max(score(a, b) for a, b in must_keep)
        assert weakest_merge < strongest_keep, (
            f"fixture: under {name} the pairs that must merge all score above every pair "
            f"that must not ({weakest_merge:.3f} vs {strongest_keep:.3f}) — a plain "
            "threshold would pass this test, which is exactly the rule that lost "
            "denorm-derived-table 2 of its 8 golden facts"
        )


async def test_the_same_claim_stated_by_two_nodes_is_reported_once():
    """(a) One claim, one statement — for a claim two NODES found, and for a claim one
    node collected from three sources."""
    _, result = await _assemble()
    body = _body(result["report_md"])

    incremental = _present(body, O_INCR, P_IVM)
    assert len(incremental) == 1, (
        "the Oracle node and the pg_ivm node state one claim — incremental maintenance "
        "applies deltas instead of rebuilding from scratch — and the report must state "
        f"it ONCE; it states it {len(incremental)} times"
    )
    tradeoff = _present(body, M_TRADE1, M_TRADE2, M_TRADE3)
    assert len(tradeoff) == 1, (
        "three sources in one node answer make the same closure-table-reads-faster / "
        f"adjacency-list-writes-faster point; the report keeps {len(tradeoff)} of them"
    )


async def test_claims_only_one_node_found_all_survive_the_merge():
    """(b) Deleting content satisfies every redundancy metric, so the merge is pinned
    against silent loss as hard as against duplication."""
    _, result = await _assemble()
    body = _body(result["report_md"])

    unique = [O_FAST, O_FULL, P_TRIG, P_IMM, M_DEF]
    lost = [KEY[s] for s in unique if s not in _present(body, *unique)]
    assert not lost, (
        f"the merge dropped {len(lost)} finding(s) no other sentence states: {lost} — a "
        "shorter report bought by deleting content is what s2_min_delta >= -5 refuses"
    )
    # ... and the duplicated claims still collapse, in the same report
    assert len(_present(body, M_TRADE1, M_TRADE2, M_TRADE3)) == 1, (
        "kept everything, merged nothing: three statements of one trade-off survived"
    )


async def test_opposite_mechanisms_on_one_topic_both_survive():
    """(f, first collapse) node 0.0.2.0: "a complete refresh ... fully recomputes the
    view" versus "an incremental refresh eliminates the need to rebuild ... from
    scratch". Same subject, same vocabulary, OPPOSITE mechanisms. The last cut merged
    the first away and golden fact 2 went with it (S2 63 -> 38)."""
    await _assert_similarity_cannot_decide()
    assert GOLDEN_COMPLETE_REFRESH.search(O_FULL), (
        "fixture: the deleted sentence must be the one that carries golden fact 2, or "
        "this test is not measuring the loss that happened"
    )
    assert not GOLDEN_COMPLETE_REFRESH.search(O_INCR) \
        and not GOLDEN_COMPLETE_REFRESH.search(P_IVM), (
        "fixture: no surviving sentence may carry that fact by accident, or the merge "
        "could delete the claim and still score"
    )

    _, result = await _assemble()
    body = _body(result["report_md"])

    assert GOLDEN_COMPLETE_REFRESH.search(body), (
        "the complete-refresh claim is gone from the report. It shares its subject and "
        "most of its vocabulary with the incremental-refresh claim, so every similarity "
        "rule collapses the two — but each answers a question the other cannot, which "
        "makes the relation OVERLAP, and overlap must never be collapsed"
    )
    assert len(_present(body, O_INCR, P_IVM)) == 1, (
        "and the genuinely equivalent pair in the same topic must still collapse — "
        "sparing everything on this topic is not a merge, it is the concatenation this "
        "stage exists to remove"
    )


async def test_a_definition_and_a_tradeoff_about_one_thing_both_survive():
    """(f, second collapse) node 0.3.1: the closure-table DEFINITION versus the
    closure-table TRADE-OFF. One concept, two claims; the last cut kept the trade-off,
    deleted the definition, and golden fact 8 went with it."""
    await _assert_similarity_cannot_decide()
    assert GOLDEN_CLOSURE_TABLE.search(M_DEF), (
        "fixture: the deleted definition must be the one that carries golden fact 8"
    )
    assert not any(GOLDEN_CLOSURE_TABLE.search(s)
                   for s in (M_TRADE1, M_TRADE2, M_TRADE3)), (
        "fixture: no trade-off sentence may carry that fact by accident, or deleting the "
        "definition would cost nothing measurable and this test would prove nothing"
    )

    _, result = await _assemble()
    body = _body(result["report_md"])

    assert GOLDEN_CLOSURE_TABLE.search(body), (
        "the closure-table definition is gone from the report, merged into a trade-off "
        "sentence about the same structure. What a closure table IS and what it COSTS "
        "are different claims: neither answers the other's question"
    )
    assert len(_present(body, M_TRADE1, M_TRADE2, M_TRADE3)) == 1, (
        "and the three restatements of the trade-off must still collapse to one"
    )


async def test_merged_claim_keeps_every_grounding_the_nodes_earned():
    """(c) After the merge every shipped [id] must still ground by the frozen scorer's
    own rule, against the page it points at — and no node may lose its citation."""
    _, result = await _assemble()
    body = _body(result["report_md"])
    citation_map = result["citation_map"]

    assert len(_present(body, M_TRADE1, M_TRADE2, M_TRADE3)) == 1, (
        "precondition for this test: a merge must have happened before asking whether "
        "it kept its grounding"
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
    assert body_ids == {ID_ROOT, ID_ORACLE, ID_PGIVM, ID_MYSQL}, (
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
    rep_ng = _ngrams(_content(result["report_md"]))

    lifted, scanned = [], 0
    for nid, node in skill.nodes.items():
        g = _ngrams(_content(node.answer_md))
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
        f"({lifted}) — that is concatenation. Exactly one is expected: the node whose "
        "wording the shared claim keeps, which is what preserves that node's grounding"
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
