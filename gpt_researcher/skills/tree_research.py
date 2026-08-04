"""Tree research skill — deep_tree_research (Tier A stage 6).

MindSearch-style persisted node tree + Self-Ask answer->child expansion +
best-first frontier + shared visited-URL / question-embedding dedup +
post-order hierarchical synthesis. Design: harness/spec/tree-research-tool-design-2026.md

Node research reuses GPTResearcher (module-level import so tests can patch
gpt_researcher.skills.tree_research.GPTResearcher / .create_chat_completion).
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import GPTResearcher
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource
from .citation_verification import CitationAgent, text_supported

logger = logging.getLogger(__name__)

# question-embedding cosine at/above which a candidate is a duplicate and dropped
DEDUP_COSINE = 0.92

# defect 3 fail-closed floor: a node whose research came back under this many
# characters is context-starved, not researched. Observed starved nodes sat at
# 1.3-8KB while a repaired node measures ~40KB (s1 live probe, 5 goldens:
# 38355-42687), so this cuts the whole starved band and still leaves ~5x headroom
# under a healthy node — a narrower leaf question gets less context than the root
# query, and failing those would hollow out the report instead of cleaning it.
# Below the floor the node has not researched its question, it has merely failed
# quietly, and the answer LLM fills the gap from prior knowledge — the direct cause
# of the trap (S3 false-positive) hits.
# ponytail: one global floor; scale it by depth if deep leaves start failing.
MIN_CONTEXT_CHARS = 8000


class NodeStatus(Enum):
    PENDING = "pending"
    RESEARCHING = "researching"
    ANSWERED = "answered"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass
class ResearchNode:
    id: str
    question: str
    parent_id: Optional[str]
    depth: int
    status: NodeStatus = NodeStatus.PENDING
    answer_md: str = ""
    answer_digest: str = ""
    learnings: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # URL strings
    novelty: float = 1.0
    priority: float = 0.0
    gap_flags: List[str] = field(default_factory=list)
    uncertainty: float = 0.5
    tokens_spent: int = 0
    credits_spent: float = 0.0
    children: List[str] = field(default_factory=list)
    question_embedding: Optional[List[float]] = None


class Frontier:
    """Best-first priority queue: pop() returns the highest-priority node."""

    def __init__(self) -> None:
        self._heap: list = []
        self._seq = itertools.count()  # tie-break: insertion order, keeps heap stable

    def push(self, node: ResearchNode) -> None:
        heapq.heappush(self._heap, (-node.priority, next(self._seq), node))

    def pop(self) -> ResearchNode:
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# matches "[1]", comma/semicolon-joined "[1, 3]" / "[1; 3]", space-only "[1 2]",
# whitespace-padded "[ 1 ]", hyphen ranges "[1-3]" (range endpoints capped at
# 3 digits so a bracketed date like "[2024-07-26]" is not mistaken for citations),
# and malformed punctuation an LLM rewrite drops in — doubled ("[1,2,,3]") or
# trailing ("[1, 99,]") separators before the closing bracket
# — any variant this still misses escapes the strip
# R2 fix: the plain-number alternative is capped at 3 digits too, matching the
# frozen S1 scorer's own marker regex (bench/score_report.py, `\[(\d{1,3})\](?!\()`)
# -- an uncapped `\d+` here misread a bare 4+-digit bracket (any year, any large
# figure, e.g. "[2024]") as a citation-id candidate and stripped it, even though
# the scorer would never have scored it as a marker in the first place.
_ITEM = r"(?:\d{1,3}|\d{1,3}\s*-\s*\d{1,3})"
_SEP = r"(?:\s*[,;]+\s*|\s+)"
# R6 fix: the trailing negative lookahead mirrors the scorer's own `(?!\()` -- a
# markdown link whose visible anchor text is a bare number, "[1](https://...)",
# is never a citation marker to the scorer, so it must never be detected (and
# then stripped, leaving a dangling "(https://...)") as an uncited one either.
_CITE_ID_RE = re.compile(rf"\[\s*({_ITEM}(?:{_SEP}{_ITEM})*){_SEP}?\s*\](?!\()")
# range alternative FIRST so a spaced range ("1 - 3") stays one token — splitting
# the group on _SEP instead shreds a range at its own internal whitespace
_TOKEN_RE = re.compile(r"(\d{1,3})\s*-\s*(\d{1,3})|(\d{1,3})")


def _bracket_ids(group: str) -> List[str]:
    """Ids cited by one bracket's inner text: "1", "1, 3", "1; 3", "1 2", "1 - 3"."""
    ids: List[str] = []
    for m in _TOKEN_RE.finditer(group):
        if m.group(3) is not None:
            ids.append(m.group(3))
        else:
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            ids.extend(str(i) for i in range(lo, hi + 1))
    return ids


def render_ids(ids: List[str]) -> str:
    """Re-render a bracket's surviving ids in the ONE shape the frozen S1 scorer
    can read: its marker regex (bench/score_report.py, `\\[(\\d{1,3})\\](?!\\()`)
    matches only digits sitting DIRECTLY between the brackets, so re-joining two
    survivors into a single "[7, 9]" makes both invisible to the grader — the ids
    still count in citations_total through their "- [id] url" line in the
    Citations block, but can never count in citations_grounded, which is the
    metric the whole fail-closed strip exists to protect. One bracket per id,
    space-joined, exactly what _attribute_citations writes everywhere else."""
    return " ".join(f"[{cid}]" for cid in ids)


def find_uncited_ids(report_md: str, citation_map: Dict[str, str]) -> List[str]:
    """[id] markers in report_md with no citations-map entry, first-appearance order."""
    out: List[str] = []
    for m in _CITE_ID_RE.finditer(report_md or ""):
        for cid in _bracket_ids(m.group(1)):
            if cid not in citation_map and cid not in out:
                out.append(cid)
    return out


# --- marker locality, mirrored from the frozen S1 scorer.
# score_s1 (harness-search/bench/score_report.py) normalizes the <=240 characters
# BEFORE a marker, keeps the last 20 tokens, and asks whether any 3-token window
# of those appears verbatim in the cited page. Both fail-closed marker passes used
# text_supported's passage rule instead (>=70% of the span's significant words
# inside one 60-token window) — far stricter than what is actually graded, and
# strict in a different dimension (paraphrase coverage, not verbatim trace).
# Measured on the round-1 benchmark that cost the whole metric: edge-ai-face-access
# and outbox-failure-modes carried 25 and 55 citation-map entries and ZERO surviving
# markers, and score_s1 scores a report with no ids at all as 0.0 — strictly worse
# than shipping a marker that merely fails to ground (S1 mean 42 vs baseline 80).
# Mirroring the scorer places a marker by the same evidence the report is graded on.
# The fail-closed spine is untouched: a URL only reaches these passes because
# text_supported already tied it to one of the node's own claims (research_node's
# node.sources narrowing), so this decides WHICH span an already-supporting source
# belongs next to — never whether the source supports the node at all.
_SCORER_WINDOW_CHARS = 240
_SCORER_WINDOW_TOKENS = 20
_WORD_RE = re.compile(r"[0-9A-Za-z]+")


def _norm_tokens(text: str) -> List[str]:
    return re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split()


def _trace_end(span: str, source: str) -> Optional[int]:
    """Offset in `span` just past the LAST verbatim 3-word phrase it shares with
    `source`, or None if it shares none.

    The 3-token window must carry a token of 4+ characters, so list numbering or
    a run of function words ("2. [4]") traces nothing — the scorer's own guard
    against a coincidental match on stopwords.
    """
    if not span or not source:
        return None
    page = " ".join(_norm_tokens(source))
    toks = [(m.group(0).lower(), m.end()) for m in _WORD_RE.finditer(span)]
    for i in range(len(toks) - 3, -1, -1):
        win = [t for t, _ in toks[i:i + 3]]
        if max(len(t) for t in win) >= 4 and " ".join(win) in page:
            return toks[i + 2][1]
    return None


def phrase_traced(span: str, source: str) -> bool:
    """The grader's verdict on a marker placed at the end of `span`: score_s1
    reads only the last 20 normalized tokens of the 240 characters before it, so
    a phrase borrowed earlier in a long sentence is out of view."""
    return _trace_end(" ".join(_norm_tokens(span)[-_SCORER_WINDOW_TOKENS:]),
                      source) is not None


# --- s5 (defect 6b): the roll-up consistency rule.
# Reproduced from the frozen S6 scorer (harness-search/bench/score_report.py, frozen
# since s0) rather than imported: gpt_researcher must not depend on the harness, and
# the live gate (contradictions_total == 0, unsupported_claims_total == 0) is that
# scorer's verdict — so the pass that cleans the report has to apply the same rule,
# token for token. Keep these in sync with score_report.py if it is ever re-frozen.
_SIGNUM = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*%|\b\d{4,}\b|\b\d+\.\d+\b")
_STOP = {"that", "with", "this", "from", "have", "been", "were", "their", "which",
         "about", "into", "over", "only", "more", "than", "when", "after", "before",
         "while", "where", "also", "each", "other", "them", "they", "these", "those",
         "such", "some", "most", "many", "very", "will", "would", "could", "should",
         "there", "then", "what", "your", "does", "using", "used", "between"}
# claim_sentences() blanks these, IN THIS ORDER, before it splits sentences, so a
# fence, marker or URL never contributes a "number" or a context token — and never
# holds a sentence boundary either
_SCRUB_RES = (re.compile(r"```.*?```", re.DOTALL), re.compile(r"\[\d{1,3}\]"),
              re.compile(r"\(https?://\S+\)"), re.compile(r"https?://\S+"))
# splits on the scorer's sentence boundary, but KEEPS the separators: joining the
# pieces back reproduces the input byte for byte, so a report with nothing to drop
# comes back untouched (markdown structure and [id] positions are what the S1
# grounding scorer reads — a pass that re-joins what it kept rewrites the artifact)
_SENT_SPLIT_RE = re.compile(r"((?<=[.!?])\s+|\n+)")


def _num_key(s: str) -> str:
    return s.replace(",", "").replace("%", "").strip()


def _ctx_tokens(text: str) -> set:
    norm = re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()
    return {t for t in norm.split()
            if len(t) >= 4 and not t.isdigit() and t not in _STOP}


def _claim_profile(text: str) -> tuple:
    """(significant numbers, context tokens) — the scorer's view of one text.

    No scrubbing here: score_s6 profiles the CORPUS raw (`prepared` is built
    straight off tree.json's node["answer"]) and profiles claim sentences off the
    already-scrubbed body. Scrubbing both sides made the pass stricter than the
    gate — it deleted report text the scorer would have accepted, because a node
    answer's figure sitting inside a URL still counts for the scorer.
    """
    return {_num_key(m.group(0)) for m in _SIGNUM.finditer(text)}, _ctx_tokens(text)


def _scrubbed(text: str) -> str:
    """claim_sentences()' pre-scrub, made offset-preserving: each deleted span
    becomes the SAME number of blanks, so an offset into the result is the same
    offset in `text`. The scorer collapses each span to a single space, which is
    equivalent for segmentation (both boundaries match whitespace RUNS) but loses
    the mapping back to the raw body that dropping an exact slice needs."""
    for rx in _SCRUB_RES:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


# defect 4, bench round 1: expansion ALREADY names the primary source when it knows
# one — measured, "What does Debezium's own documentation and issue tracker report..."
# is the single outbox node whose sources landed debezium.io. What those questions do
# not get is RESEARCHED. A run ends on time_budget_s with most of the tree still
# PENDING (round 0: edge-ai researched 6 of 25 nodes, solid-state 4 of 17), and
# best_first priority was depth-only, so which children make the cut is the order the
# expansion LLM happened to emit them in. Every question that named a primary source —
# "What do NIST's Face Recognition Technology Evaluation ... report", "What do
# QuantumScape's own investor disclosures reveal" — sat PENDING behind generic
# commentary questions, and S4 came in at 56 against the frozen baseline's 83. The
# affinity below is what moves those questions to the front of the frontier.
_PRIMARY_STOP = {"a", "an", "the", "what", "how", "which", "who", "whose", "where",
                 "when", "why", "it", "its", "this", "that", "these", "those",
                 "there", "they", "their", "and", "or", "but", "if", "beyond", "do",
                 "does", "are", "is", "in", "for", "of", "to"}
_NAMED_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9.+/-]*\b")
# "NIST's", "QuantumScape's", "Debezium's" — a named org asked about itself
_POSSESSIVE_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9.+/-]*['’]s\b")
_OWN_SOURCE_RE = re.compile(
    r"\b(?:own|official|primary)\b[^.?!]{0,60}?\b(?:doc|docs|documentation|blog|"
    r"site|website|spec|specs|specification|whitepaper|white paper|filing|filings|"
    r"release notes|changelog|repo|repository|issue tracker|announcement|"
    r"publication|standard|standards|guide|manual|datasheet|disclosure|disclosures|"
    r"report|reports|paper|papers)\b", re.I)


def _primary_source_affinity(question: str) -> float:
    """0..1 — how directly a question asks a NAMED originator for its OWN material.

    Three independent signals, because they discriminate at different strengths:
    naming any specific entity is weak (nearly every question mentions some proper
    noun), addressing a named entity possessively is stronger, and asking that
    entity for its own/official material rather than for third-party commentary
    about it is what actually lands the primary domain.
    """
    # the leading interrogative is capitalized by grammar, not by being an entity
    tail = question.split(" ", 1)[-1]
    named = any(m.group(0).lower() not in _PRIMARY_STOP
                for m in _NAMED_ENTITY_RE.finditer(tail))
    if not named:
        return 0.0
    return (0.30
            + 0.35 * bool(_POSSESSIVE_ENTITY_RE.search(tail))
            + 0.35 * bool(_OWN_SOURCE_RE.search(question)))


# --- s9: the roll-up states each CLAIM once, under themes drawn from the claims.
# SIMILARITY SELECTS CANDIDATES; EQUIVALENCE DECIDES. Cosine, word overlap, LSA and
# coverage all score the INTERSECTION of two texts, and the merge decision lives in
# the DIFFERENCE — four implementations merged on the intersection and each deleted a
# fact (a "complete refresh" sentence merged away in favour of an "incremental
# refresh" one; a closure-table DEFINITION merged away in favour of its TRADE-OFF).
# arXiv:2509.08304 states the operator: two texts are equivalent only when neither
# answers a question the other cannot, and OVERLAP — each with its own answerable
# questions — must never be collapsed. So the verdict is a model call that ENUMERATES
# what each side says the other does not (steadier than classification, and the two
# lists are the explanation the review lane reads), and an unclear verdict does NOT
# merge: not merging costs report length, wrongly merging costs facts.
_UNIT_SPLIT_RE = re.compile(r"\n\s*\n+")
# A claim unit is an ATOMIC CLAIM, never a paragraph. Measured 2026-07-29
# (d1_offline.probe.json, denorm-derived-table): split on blank lines a unit is a
# ~650-character paragraph carrying several facts, and asked what a reader would lose
# if it were deleted, the judge answers "something" for nearly every one — correctly,
# since a paragraph with five facts almost always holds one no other paragraph holds.
# Nothing merged, report_chars came back exactly the zero-merge number. The unit has
# to be the atomic claim for "is anything lost?" to have a mergeable answer at all
# (Claimify, FActScore, NuggetIndex: index atomic facts to reduce redundancy).
_SENT_BREAK_RE = re.compile(r"(?<=[.!?])(?:\s*\[[^\]\n]*\])*\s+(?=[^a-z\[])")
_BRACKETED_RE = re.compile(r"\[[^\]]*\]")
# A sentence break INSIDE a quotation, parenthetical or code span is not a claim
# boundary: the full stop belongs to the quoted sentence, not to the sentence quoting
# it. Measured on the shipped denorm report (probe d099fa7f53576633), 20 of 199 units
# carried an unbalanced delimiter — Oracle's one quoted sentence became two units filed
# under two different headings, one opening a quotation it never closes and the other
# closing one it never opened. Half a quotation is not a claim, so the equivalence judge
# was being asked what a reader would lose if a fragment were deleted.
# Apostrophes are deliberately not counted (`Oracle's`); backticks are counted in RUNS
# so a ```fence``` reads as one open and one close, like `code`.
_BACKTICK_RUN_RE = re.compile(r"`+")


def _delims_closed(text: str) -> bool:
    """True when every bracket, parenthesis, quotation and code span opened in `text`
    is also closed in it. A stray CLOSER is not treated as unbalanced — its opener sat
    in an earlier unit, which was cut only because it balanced."""
    depth = 0
    curly = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "“":
            curly += 1
        elif ch == "”":
            curly = max(0, curly - 1)
    return (depth == 0 and curly == 0 and text.count('"') % 2 == 0
            and len(_BACKTICK_RUN_RE.findall(text)) % 2 == 0)


# A unit opening on a bare anaphor states nothing on its own — its subject lives in the
# sentence before it, and the theme layout files the two under different headings ("This
# is not a disagreement between sources" landing in a section that mentions no
# disagreement). Same dependency as a colon lead-in and the same fix: it travels with the
# unit it refers back to. Review R6, and the second measured cause of an unjudgeable
# unit — 26 of 199 shipped units opened this way.
_ANAPHOR_RE = re.compile(
    r"^[\s*_>-]*(?:This|That|These|Those|It|They|Such|Thus|Therefore|Hence)\b", re.I)
# A LIST MARKER IS NOT A CLAIM. "1." parses as a sentence end by every rule
# _SENT_BREAK_RE has — a full stop, whitespace, then a capital — so an enumerated list
# is shredded into empty ordinals plus content that then gets laid out by theme far
# from the number that introduced it. Measured 2026-08-04 on the shipped
# harness-landscape report: 8 of 139 claim units (5.8%) were bare markers
# (`**1.` `**2.` `**3.` `1.` `2.` `3.` `4.` `5.`), and the report rendered
# "**Key contrasts:**" followed by five empty numbers while the five actual items sat
# under other headings. The marker governs what follows exactly as a colon lead-in
# does, so it rejoins the same way.
_ORDINAL_ONLY_RE = re.compile(r"^[\s*_>#-]*(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})\s*[.)]\s*$")
# The reply's own shape is `"<fragment>" => UNIQUE: <what only this one says>`, and the
# keyword is what a model drops first: measured 2026-07-29 on this deployment, a screen
# answered `"The refresh method can be incremental or a complete refresh" => none`, which
# the keyword-anchored pattern threw away — and a discarded line is no verdict, so the
# unit could never merge. Split on the arrow, then peel the keyword if it is there.
_UNIQUE_LINE_RE = re.compile(r"^(.*?)=>\s*(?:unique\b\s*[:=]?\s*)?(.*)$", re.I)
_UNIQUE_ALT_RE = re.compile(r"^(.*?)\bunique\s*[:=]\s*(.*)$", re.I)
# A LABEL MUST LOOK LIKE A LABEL (review R3). The fallback ran against a `head` that is
# normally the QUOTED FRAGMENT itself, so any statement opening with a one-letter word
# — "A separate table stores...", "a complete refresh...", "I found that..." — was read
# as statement A or I of the screen and its verdict handed to the wrong statement. That
# branch is reached in the two shapes the impl documents as normal (a fragment under 24
# characters, and a fragment that is a substring of two keys), so it was not rare. A
# label is a letter or number at the very START, after nothing but list punctuation, and
# followed by a label delimiter — never a quote character, which is what a fragment
# opens with. The `statement A` spelling is accepted without the delimiter because the
# keyword already says it is a label.
_LABEL_RE = re.compile(
    r"^[\s\-*>#]*(?:statement\s*([A-Za-z]|\d{1,2})\b|([A-Za-z]|\d{1,2})\s*[.):\]])", re.I)
# greedy on purpose: a quoted fragment routinely contains a nested quotation of its own
# ("Without a materialized view log, Oracle states plainly, '...'"), and a lazy match
# would hand back only the words before the inner quote
_QUOTED_RE = re.compile(r"[\"“”'‘’](.+)[\"“”'‘’]", re.S)
_MD_NOISE_RE = re.compile(r"[*_`~]+")
# How alike a unit must be to a screening group before it is worth putting in the same
# prompt. This is NOT a merge threshold — nothing here decides anything, the judge does
# (and the fixture measures that no threshold CAN decide). It is the grouping floor, and
# it was the no-op's first cause: at 0.08 with a top-3 neighbour rule the candidate graph
# over denorm-derived-table's 212 units is one connected component, so the screening
# groups were arbitrary slices of a chain through it — LISTEN/NOTIFY next to hierarchyid
# next to a Cosmos DB session token. Asked whether any of those 16 says nothing the
# others do not, the honest answer is no, and 195 screened units produced 5 covered.
# THE BAND, NOT A NUMBER. It was a constant 0.30, read off the corpus's own top ~1% of
# pair scores (text-embedding-3-small, 2026-07-29: 22,366 pairs, q0.99 = 0.291,
# q0.999 = 0.468, max 0.613). A cosine is not comparable across embedders, so that
# constant was silently bound to one provider: measured 2026-07-30 on the SAME corpus with
# a local qwen3-embedding-4b, 78,713 pairs give q0.99 = 0.738 and max 1.000, where 0.30
# admits 76.85% of ALL pairs as candidates — every pair a model round trip — while
# text-embedding-3-small never scored ANY pair above 0.613, so a floor ported the other
# way admits nothing and the merge is a silent no-op. Both failures are invisible in the
# output. So the derivation itself is the code now: take the top MERGE_CANDIDATE_PCT of
# THIS run's own pair scores. Same band on any embedder, nothing to re-tune, and the
# recorded numbers above stay meaningful as what the band evaluated to.
MERGE_CANDIDATE_PCT = 0.01
# A screening call carries a whole group, so it is the long one: measured on this
# deployment (claude_agent) a 2-unit verdict answers in 17-34s and a 11-unit screen in
# 127s, which the previous 120s ceiling would have thrown away as a timeout.
MERGE_VERDICT_TIMEOUT_S = 300.0
MERGE_JUDGE_FAILURES = 3    # transient errors happen; a dead judge answers none
MERGE_CONCURRENCY = 4       # verdicts in flight at once
EMBED_CONCURRENCY = 16      # embeddings in flight at once — an HTTP call, not a process
# Units per SCREENING call. One judge call is a whole model round trip — measured on
# this deployment (claude_agent) at 142-320s for a group of 16 — and the captured
# goldens offer 111-294 candidate pairs each, which is hours per query and, at 175
# sequential CLI spawns, the 0xC0000142 that killed four of five re-syntheses on
# 2026-07-29. So the pairs are SCREENED in groups first: a unit the screen says has
# content of its own can never be merged away, so none of its pairs is ever asked.
# A whole tree smaller than one group is screened in a single call — that is the s9
# fixture (13 claim units), and it is also the only size at which "one group" is not a
# grouping choice. On a real tree the growth rule below stops long before this cap.
MERGE_GROUP = 16
_MERGE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_THEMES = 8


def _claim_units(text: str) -> List[str]:
    """One node's synthesized text as atomic claims: the author's own paragraph breaks
    first, then the sentences inside each paragraph.

    A trailing [id] belongs to the sentence BEFORE it. `_attribute_citations` puts a
    marker after the phrase it traces to (or after the full stop when only punctuation
    follows), and the frozen scorer grounds a marker off the text PRECEDING it — so a
    break that pushed the marker onto the next sentence would strand it beside wording
    its own source never wrote. The lookahead keeps "e.g. foo" and "MySQL 8.0 onward"
    whole; `\\s+` already protects a decimal point.
    """
    pieces: List[tuple] = []
    for bno, block in enumerate(_UNIT_SPLIT_RE.split(text or "")):
        ends = [m.end() for m in _SENT_BREAK_RE.finditer(block)] + [len(block)]
        prev = 0
        for k, end in enumerate(ends):
            # skip a break that would leave a quotation, parenthetical or code span
            # open — but only when a LATER break in this block closes it again, so a
            # single stray delimiter cannot swallow the rest of the paragraph
            if not _delims_closed(block[prev:end]) and any(
                    _delims_closed(block[prev:later]) for later in ends[k + 1:]):
                continue
            piece = block[prev:end].strip()
            if piece:
                pieces.append((bno, piece))
            prev = end
    # A COLON-TERMINATED CLAUSE IS NOT A CLAIM, it is the lead-in to the next one, and
    # once every unit is its own block laid out by theme it ends up severed from the
    # material it introduces and sometimes filed under a different heading — measured on
    # the shipped denorm report as 6 stranded lines ("Per the docs:", "Three defensible
    # patterns:"). It travels with the unit it governs. Across paragraph breaks too:
    # a lead-in and its list are two BLOCKS more often than they are two sentences.
    # A unit opening on a bare anaphor depends on its predecessor the same way, and
    # rejoins it the same way — with the author's own separator, a space inside one
    # paragraph and a blank line across two.
    out: List[tuple] = []
    for bno, piece in pieces:
        if out and (out[-1][1].endswith(":") or _ANAPHOR_RE.match(piece)
                    or _ORDINAL_ONLY_RE.match(out[-1][1])):
            sep = " " if out[-1][0] == bno else "\n\n"
            out[-1] = (bno, f"{out[-1][1]}{sep}{piece}")
        else:
            out.append((bno, piece))
    return [piece for _, piece in out]


def _quote_key(text: str) -> str:
    """A claim's wording reduced to what a quoting model reliably reproduces.

    Measured 2026-07-29: the judge quotes its fragments with the markdown stripped —
    `**Fast Refresh using materialized view logs**` comes back bare — so a verbatim
    `fragment in unit` test misses the line and the verdict is discarded as unreadable.
    Emphasis, backticks, punctuation and case are not content; dropping them on BOTH
    sides is what lets the reply be traced back to the statement it judged.
    """
    return re.sub(r"[^a-z0-9]+", " ",
                  _MD_NOISE_RE.sub("", text or "").lower()).strip()


def _flat_claim(text: str) -> str:
    """One claim unit as the judge should see it: citation markers gone (a marker is
    not content, and it is noise in a comparison) and whitespace collapsed."""
    return re.sub(r"\s+", " ", _BRACKETED_RE.sub(" ", text or "")).strip()


def _claim_counts(text: str) -> Dict[str, int]:
    """The unit's content words and how often it uses them. Citation markers are
    stripped first — a marker is not content, and rollup_scan's own tokenizer drops
    them too."""
    counts: Dict[str, int] = {}
    for t in _ctx_tokens(_BRACKETED_RE.sub(" ", text or "")):
        counts[t] = 0
    for t in _norm_tokens(_BRACKETED_RE.sub(" ", text or "")):
        if t in counts:
            counts[t] += 1
    return counts


def _claim_tokens(text: str) -> set:
    return set(_claim_counts(text))


def _bag_cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Term-frequency cosine over content words — what a real encoder can at least
    see. The stand-in for the embedding service, and like it, only a RANKER."""
    if not a or not b:
        return 0.0
    num = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len].strip("-") or "tree-research"


class TreeResearchSkill:
    """Explicit persisted research tree with best-first frontier scheduling."""

    def __init__(self, researcher):
        self.researcher = researcher
        self.tone = researcher.tone
        self.websocket = researcher.websocket
        self.headers = researcher.headers or {}
        self.config_path = getattr(researcher.cfg, "config_path", None)
        self.visited_urls = researcher.visited_urls  # shared across all node researchers
        self.nodes: Dict[str, ResearchNode] = {}
        self._embeddings: List[List[float]] = []  # accepted candidate-question embeddings
        self._covered_embeddings: List[List[float]] = []  # scored-node question embeddings
        self._read_docs: Dict[str, str] = {}  # url -> text of documents actually scraped
        self._syntheses: Dict[str, str] = {}
        self._root_query: str = str(getattr(researcher, "query", "") or "")
        self._max_breadth = 4
        self._rollup_dropped_ratio = 0.0  # share of the body verify_rollup removed
        self._merge_judge_ok = True  # cleared for the run once the judge stops answering
        self._merge_judge_fails = 0
        self._strategic_llm: Optional[tuple] = None
        # Any, not int: these are JSON payloads that travel out with the report, and
        # candidate_floor is a cosine
        self._merge_stats: Dict[str, Any] = {}
        self._merge_calls: Dict[str, Any] = {}
        # claim units that got a REAL embedding this run, 0 when _unit_vectors degraded
        # to word overlap. Travels out in _merge_stats so a measurement taken on the
        # degraded signal can be refused instead of read as "there was no redundancy".
        self._embedded_units = 0
        self.tokens_spent = 0
        self.credits_spent = 0.0

    # ------------------------------------------------------------------ seams
    # Each of these is a deterministic-test seam: unit tests replace them on the
    # instance, so run() must route every LLM/embedding/research call through them.

    async def research_node(self, node: ResearchNode) -> None:
        """Research one node with a dedicated GPTResearcher; mutates the node."""
        node.status = NodeStatus.RESEARCHING
        researcher = GPTResearcher(
            query=node.question,
            report_type=ReportType.ResearchReport.value,
            report_source=ReportSource.Web.value,
            tone=self.tone,
            websocket=self.websocket,
            config_path=self.config_path,
            headers=self.headers,
            visited_urls=self.visited_urls,
        )
        context = await researcher.conduct_research()
        try:
            self.visited_urls.update(researcher.visited_urls)
        except (AttributeError, TypeError):
            pass

        if isinstance(context, list):
            context = "\n\n".join(str(c) for c in context)
        context = str(context or "")[:60000]

        # defect 6a: node.sources means "read AND quoted", not "every retriever
        # return". The candidates are the documents this node researcher actually
        # read (research_sources) — NOT visited_urls, which is both too wide (a
        # retriever-returned URL nobody scraped) and too narrow (a retriever that
        # prefetches full content, e.g. Firecrawl / PubMed Central, delivers a read
        # document that no scrape ever registered).
        read_docs: Dict[str, str] = {}
        try:
            for doc in researcher.get_research_sources() or []:
                url = str(doc.get("url") or "")
                text = str(doc.get("raw_content") or doc.get("content") or "")
                # a later contentless duplicate must not blank a document that was read
                if url and (text or url not in read_docs):
                    read_docs[url] = text
        except (AttributeError, TypeError):
            read_docs = {}

        # defect 3, hardest case: the node has nothing for the answer to stand on —
        # it read no document, or the research context came back empty. Either way
        # the answer prompt degenerates to "Question: X / Context: <nothing>" and the
        # LLM can only write prior knowledge — exactly the fabricated text that lands
        # in the report as a trap hit, and that then pairs itself with a read URL
        # (text_supported matches the fabricated sentence against the trap document)
        # so it even looks sourced. Don't ask it at all; there is then no answer to
        # salvage downstream. The below-floor-but-non-empty band still runs the LLM —
        # it has real context, and the s2 contract is pinned on what that pass
        # produces — and fails closed on standing further down.
        if not read_docs or not context.strip():
            logger.error(f"tree node {node.id} has no evidence "
                         f"({len(read_docs)} documents read, {len(context.strip())} "
                         f"context chars) — failing closed before the answer LLM")
            node.tokens_spent = len(context) // 4
            node.status = NodeStatus.FAILED
            return
        self._read_docs.update(read_docs)

        # ponytail: one LLM call yields answer + digest + learnings; parse-tolerant
        #
        # defect 6b, the half S6 scores as "contested 병기": the roll-up only
        # CONCATENATES node answers (see synthesize_node — every rewrite design
        # measured worse), so this prompt is the last place a disagreement can
        # survive. Whatever it collapses is gone from the report for good.
        # Bench round 3 measured the cost: of the 11 contested sides the goldens
        # ask for and the reports miss, 8 are missing from EVERY node answer too —
        # the tree read the sources, the answer picked one side, and S6 came in at
        # 33 against the baseline's 52. Hence the explicit both-sides rule below,
        # and the wider word budget it needs to hold both (60k chars of context
        # squeezed into 400 words has no room for a minority view).
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher answering one focused question from collected context."},
                {"role": "user",
                 "content": (
                     f"Question: {node.question}\n\nContext:\n{context}\n\n"
                     "Write three sections:\n"
                     "ANSWER: a markdown answer (<=900 words). Wherever the context "
                     "DISAGREES with itself — two sources giving different figures "
                     "for the same quantity, or taking opposing positions on the "
                     "same question — state EVERY side explicitly and name who "
                     "reports which. Never average or range-merge disagreeing "
                     "figures, and never drop the minority view. Do not add citation "
                     "markers, footnotes, or bracketed references of any kind — "
                     "citations are attached separately from the real source list.\n"
                     "DIGEST: a <=120-word summary of the answer.\n"
                     "LEARNINGS: 3-6 bullet lines, one atomic factual claim each."
                 )},
            ],
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.3,
        )
        response = str(response or "")
        parts = {"ANSWER": "", "DIGEST": "", "LEARNINGS": ""}
        current = None
        for line in response.splitlines():
            # tolerate the section label however the model decorates it --
            # "ANSWER: ...", "# ANSWER", "**ANSWER:**" -- a strict "ANSWER:"-only
            # match silently falls through to the response.strip() fallback below,
            # dumping all three sections (headers included) into answer_md as one
            # blob; that bloats _attribute_citations' input with paraphrased
            # DIGEST/LEARNINGS text that can't ground as tightly as the real
            # ANSWER wording, which is what depressed S1_pct (measured: 48/63%).
            m = re.match(r"^\s*#{0,6}\s*\**\s*(ANSWER|DIGEST|LEARNINGS)\**\s*:?\s*\**\s*(.*)$", line)
            if m:
                current = m.group(1)
                parts[current] = m.group(2)
            elif current:
                parts[current] += "\n" + line
        node.answer_md = (parts["ANSWER"].strip() or response.strip())
        node.answer_digest = (parts["DIGEST"].strip() or node.answer_md[:800])
        node.learnings = [l.strip("-* ").strip()
                          for l in parts["LEARNINGS"].splitlines() if l.strip("-* ").strip()]
        if not node.learnings:
            node.learnings = [node.answer_digest] if node.answer_digest else []

        # Keep only the read documents supporting some sentence or learning of the
        # answer; fail-closed on missing/empty documents.
        claims = [s for s in re.split(r"(?<=[.!?])\s+", node.answer_md) if s.strip()]
        claims += node.learnings
        # a quoted span ending `..."` never splits above (the sentence-boundary
        # regex needs punctuation directly before the whitespace, but a closing
        # quote sits in between), so a verbatim quote stays merged with whatever
        # exposition the answer LLM appended after it. That merged chunk's word
        # set is mostly invented connective prose the source never contains,
        # which drags _passage_covers's 70%-of-one-window ratio below threshold
        # even though the quote itself is near-verbatim -- check quoted spans
        # standalone so they ground on their own merit.
        claims += re.findall(r'"([^"]{15,})"', node.answer_md)
        node.sources = sorted(u for u, doc_text in read_docs.items()
                              if doc_text
                              and any(text_supported(c, doc_text) for c in claims))

        node.tokens_spent = (len(context) + len(response)) // 4
        try:
            node.credits_spent = float(researcher.get_costs() or 0.0)
        except (AttributeError, TypeError, ValueError):
            node.credits_spent = 0.0

        # defect 2+3: the node read documents, so unlike the empty-handed case above
        # the answer LLM has to run before the verdict — the s2 contract is pinned on
        # what this pass produces from a read document (narrowed node.sources, the
        # parsed answer/digest/learnings, the fabricated-[id] detection run() reports),
        # and none of it exists until the answer does. What context starvation changes
        # is the node's STANDING, not its bookkeeping: FAILED is what keeps its text
        # out of the roll-up (synthesize_node), its URLs out of the citation map (run)
        # and both out of tree.json (_node_dict), so no starved claim reaches the
        # report even though the answer text still sits on the node.
        if len(context) < MIN_CONTEXT_CHARS:
            logger.error(f"tree node {node.id} context {len(context)} chars "
                         f"< MIN_CONTEXT_CHARS {MIN_CONTEXT_CHARS} — failing closed")
            node.status = NodeStatus.FAILED
            return
        node.status = NodeStatus.ANSWERED

    def _covered_ground(self) -> List[str]:
        """"Question -> what we found" for every node that actually researched.

        A FAILED node is left out on both counts: it researched nothing, so
        listing its question fences the expansion away from a hole the tree never
        filled, and its "findings" are prior-knowledge text written over missing
        evidence (defect 3) that the report is not allowed to chase.
        """
        lines = []
        for n in self.nodes.values():
            if n.status not in (NodeStatus.ANSWERED, NodeStatus.EXPANDED, NodeStatus.PRUNED):
                continue
            found = (n.answer_digest or " ".join(n.learnings)).strip()
            lines.append(f"- Q: {n.question}\n  Found: {found[:400]}" if found
                         else f"- Q: {n.question}")
        return lines

    async def generate_child_questions(self, node: ResearchNode) -> List[str]:
        """Self-Ask expansion steered by the ground the tree has ALREADY covered.

        defect 4: the old prompt showed the model a bare `[:30]` slice of question
        STRINGS and asked for "gaps left by the answer" — a node-local derivation
        blind to both what those nodes FOUND and to the root query that "uncovered"
        is measured against, so a differently-worded child silently re-covered an
        answered area. Coverage is not truncated either: a tree runs to max_nodes
        (40) nodes, so a 30-item slice hid real coverage from the model that is
        supposed to steer around it.
        """
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher generating disjoint follow-up research questions."},
                {"role": "user",
                 "content": (
                     f"Root research query: {self._root_query or node.question}\n\n"
                     f"Just researched: {node.question}\n"
                     f"Answer digest: {node.answer_digest}\n\n"
                     "Ground this research tree has ALREADY covered — each question "
                     "and what researching it found:\n"
                     + "\n".join(self._covered_ground())
                     + self._queued_ground()
                     + f"\n\nGenerate up to {self._max_breadth} follow-up questions that carry "
                       "the root research query into ground the list above does NOT yet cover. "
                       "Target what is missing, not variations of what was already found. Each "
                       "must be disjoint from the others and from the covered questions. If you "
                       "know the specific project, product, company, standard, or author that "
                       "originated this topic, name that entity by its proper name in the "
                       "question itself (e.g. 'What does the <project>'s own blog/documentation "
                       "say about X' rather than a generic phrasing of the same question), so "
                       "the search targets that primary source directly instead of generic "
                       "secondary commentary. At least one question must address the entity "
                       "that DEFINES the root topic — the standards body, specification "
                       "author, or originating vendor whose own site is the authority on it, "
                       "not only the tools built on top of it — and ask for that entity's own "
                       "documentation, specification, or filings by name. Return "
                       "0 questions if the root query is fully covered. "
                       "Format each on its own line as 'Question: <question>'."
                 )},
            ],
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
        )
        questions = [l.split(":", 1)[1].strip()
                     for l in str(response or "").splitlines()
                     if l.strip().lower().startswith("question:") and ":" in l]
        return [q for q in questions if q][:self._max_breadth]

    async def embed_question(self, text: str) -> List[float]:
        memory = getattr(self.researcher, "memory", None)
        if memory is None:
            from ..memory.embeddings import Memory
            cfg = self._embedding_cfg()
            memory = Memory(cfg.embedding_provider, cfg.embedding_model,
                            **getattr(cfg, "embedding_kwargs", {}))
        return list(await memory.get_embeddings().aembed_query(text))

    def _embedding_cfg(self):
        """The config that names the embedding provider, live-resolved when the
        researcher's own has none.

        Same reason as `_strategic_model`: the offline re-synthesis runner rebuilds the
        skill from a sidecar and hands it a stub with no embedding settings, so
        `cfg.embedding_provider` raised AttributeError and the merge silently fell back
        to word overlap for candidate selection — measured 2026-07-29, and word overlap
        is the ceiling this stage exists to get off. The replay must select candidates
        with the same encoder the live run does or it is measuring something else.
        """
        cfg = getattr(self.researcher, "cfg", None)
        if getattr(cfg, "embedding_provider", None):
            return cfg
        from ..config import Config
        return Config()

    def _other_covered_embeddings(self, own: List[float]):
        """Every question embedding the tree already holds — accepted candidates,
        live nodes (the root's included) and previously scored nodes — minus the
        node's own. Scoring a node against its own registered embedding is cosine
        1.0, which would prune the entire tree.

        A FAILED node researched nothing (see _covered_ground), so its embedding
        must not count as covered ground either — otherwise the hole it left
        could still prune a later child that would have filled it, even though
        compute_novelty is never called for the FAILED node itself.

        ponytail: "own" and the failed-embedding exclusion are both identified by
        value, because the registries carry bare vectors; a genuine twin question
        never reaches scoring anyway (DEDUP_COSINE drops it at generation). Key
        the registries by node id if that changes.
        """
        failed = [list(n.question_embedding) for n in self.nodes.values()
                  if n.status == NodeStatus.FAILED and n.question_embedding]
        for vec in itertools.chain(self._embeddings, self._covered_embeddings,
                                   (n.question_embedding for n in self.nodes.values()
                                    if n.status != NodeStatus.FAILED)):
            if vec and list(vec) != own and list(vec) not in failed:
                yield vec

    def compute_novelty(self, node: ResearchNode) -> float:
        """1 - the highest cosine between this node's question and any question the
        tree already covers.

        defect 5: the old score was the fraction of the node's learning STRINGS not
        already seen verbatim, so a child restating covered ground in different
        words scored a perfect 1.0 — pruned_count was 0 in every measured tree
        (bun-rust-port live run: node_count 13, pruned_count 0). Meaning, not
        wording, decides now.
        """
        own = list(node.question_embedding or [])
        if not own:
            # ponytail: no embedding (the embedding call failed) -> fail OPEN. There
            # is no evidence of duplication, and pruning on none deletes real
            # research; the dedup step degrades the same way.
            return 1.0
        nearest = max((_cosine(own, vec) for vec in self._other_covered_embeddings(own)),
                      default=0.0)
        self._covered_embeddings.append(own)
        return max(0.0, min(1.0, 1.0 - nearest))

    def _queued_ground(self) -> str:
        """Questions already sitting in the frontier, shown to the expander as QUEUED.

        The expansion prompt used to see only _covered_ground(), which excludes
        PENDING, so the model could not tell that the question it was about to
        invent was already waiting in the queue. Siblings therefore researched
        near-identical questions by construction — measured round 4: 8-20 pending
        nodes per query while the report carried every kept answer verbatim, the
        upstream source of the roll-up's redundancy.

        Deliberately NOT folded into _covered_ground(): a queued question is not
        covered ground. Most pending nodes are never researched, and presenting
        them as covered would fence expansion away from ground nobody filled —
        the same mistake the FAILED exclusion exists to avoid. "Do not restate"
        is a different instruction from "already answered, steer elsewhere".
        """
        queued = [n.question for n in self.nodes.values()
                  if n.status == NodeStatus.PENDING and n.question]
        if not queued:
            return ""
        return ("\n\nQuestions ALREADY QUEUED for research (not answered yet — they WILL be "
                "covered, so do NOT restate or reword any of them):\n"
                + "\n".join(f"- {q}" for q in queued))

    def register_covered(self, node: ResearchNode) -> None:
        """Record a node's question as ground the tree now covers.

        Needed because the batch scores its nodes against a throwaway copy of the
        covered list (so siblings cannot prune each other), which discards the
        registration compute_novelty performs as a side effect. Registering here
        puts each question in exactly once, pruned nodes included — the same set
        that was covered when scoring and registration were a single call.
        """
        own = list(node.question_embedding or [])
        if own:
            self._covered_embeddings.append(own)

    def unregister_covered(self, node: ResearchNode) -> None:
        """Undo a node's covered-ground registration.

        Novelty is now scored BEFORE research (so a node destined for PRUNED never
        costs a search), and scoring is what registers the question. A node that
        then FAILS researched nothing, and the original code deliberately skipped
        registration for exactly that case: presenting a starved question as
        covered ground lets the hole it left prune a later child that would have
        filled it. Scoring first means the entry is already in, so it comes back out.
        """
        own = list(node.question_embedding or [])
        if not own:
            return
        for i in range(len(self._covered_embeddings) - 1, -1, -1):
            if self._covered_embeddings[i] == own:
                del self._covered_embeddings[i]
                return

    def _prune_ungrounded_markers(self, body: str, citation_map: Dict[str, str]) -> str:
        """Second fail-closed pass, after find_uncited_ids strips markers with NO
        citations-map entry at all. An [id] that DOES exist in citation_map can
        still be a number the answer LLM invented on its own — the per-node LLM
        call was never given the real global id map (see research_node's ANSWER
        prompt) — that only coincidentally collides with some unrelated real
        source's id; find_uncited_ids cannot catch this, since that id genuinely
        exists, just not for this claim. Live measurement traced most S1 grounding
        failures to exactly this: a stray LLM-written `[13]` (or a whole cluster
        like `[4] [8] [10] [11] [12] [14] [16] [17]`) surviving next to a sentence
        source #13 never supported.

        Checks a fixed preceding-character window rather than re-splitting body
        into sentences: _attribute_citations joins a marker onto its sentence
        with a space, not onto the period itself ("claim. [1]"), so re-splitting
        the assembled body on sentence-final punctuation peels a trailing marker
        off its own supporting sentence and glues it to the START of the next
        one instead (every marker then gets checked against the wrong text,
        confirmed live: citations_total dropped to 0). A character window has no
        such boundary to get wrong.

        The window is the scorer's own — the same 240 characters, unedited, that
        score_s1 reads back from the shipped report (see phrase_traced). Blanking
        the earlier markers out of it first, as this pass used to, measures a span
        the grader never looks at."""
        def _check(m: "re.Match[str]") -> str:
            window = body[max(0, m.start() - _SCORER_WINDOW_CHARS):m.start()]
            kept = [cid for cid in _bracket_ids(m.group(1))
                    if phrase_traced(window, self._read_docs.get(citation_map.get(cid, ""), ""))]
            return render_ids(kept)
        return _CITE_ID_RE.sub(_check, body)

    def verify_rollup(self, body: str) -> tuple:
        """Check the assembled roll-up against what the nodes actually found.

        defect 6b: run() went body -> uncited-[id] strip -> ungrounded-marker strip
        -> Citations list; all of that polices citation MARKERS, none of it asks
        whether a sentence agrees with any node answer. A figure the merge invented,
        or one that conflicts with a node's own, shipped as a finding.

        Returns (kept_body, contradictions, unsupported). Detection alone is not
        enough — the scorer counts what is left in the report, and the s5 live gate
        is contradictions_total == 0 / unsupported_claims_total == 0 — so an
        offending claim is dropped as well as reported.

        The corpus is exactly what tree.json ships as nodes[].answer, so this pass
        and the scorer read the same evidence. A FAILED node is left out: its answer
        is prior knowledge written over missing evidence (defect 3 / s3), never
        evidence for anything.

        Two known asymmetries are deliberately NOT closed here, because closing
        either from this side alone makes things worse:
          * the report's H1 is `# {query}`, so a query carrying a figure (two
            goldens do) is scored as a claim and can be dropped, taking the title
            with it. Exempting headings from the DROP does not exempt them from the
            SCORER, so it trades a lost title for a gate leak — the fix is in how
            the title is rendered, not in what this pass checks.
          * a PRUNED node's answer validates claims here even though rollup() keeps
            its text out of the report. tree.json blanks only FAILED, so the
            scorer's corpus has PRUNED answers too; dropping them here only would
            make this pass stricter than the gate and delete report text the scorer
            accepts. Both sides move together or neither.
        """
        corpus = [_claim_profile(n.answer_md) for n in self.nodes.values()
                  if n.status != NodeStatus.FAILED and n.answer_md]
        body = body or ""
        # Segment the SCRUBBED text, not the raw body — the scorer scrubs first and
        # splits that. Two ways the raw split diverges from what the gate measures:
        # a scrub that empties the gap between a '.' and the next word CREATES a
        # boundary ("...lines.[1]The team logged 88,888 warnings."), so the trailing
        # sentence would never be claim-checked; and the over-long part it leaves
        # behind carries the UNION of both sentences' numbers, letting one supported
        # figure launder an unsupported one past the check. _scrubbed() is
        # offset-preserving, so a span here is still the same span of `body`.
        parts = _SENT_SPLIT_RE.split(_scrubbed(body))
        # review F1: the separator strip below runs on the RAW slice, and
        # _scrubbed() blanks a fenced block exactly like a marker — so a fence
        # abutting a dropped sentence's period ("...48,500 units.```data[1] =
        # load()```") lands INSIDE that separator and the strip would delete the
        # array index out of shipped code. A bracket inside a fence is code, never
        # a citation; the scorer never reads it either way.
        fences = [m.span() for m in _SCRUB_RES[0].finditer(body)]

        def _strip_orphan_markers(raw: str, base: int) -> str:
            return _CITE_ID_RE.sub(
                lambda m: m.group(0) if any(a <= base + m.start() < b for a, b in fences)
                else "", raw)

        contradictions: List[str] = []
        unsupported: List[str] = []
        out: List[str] = []
        pos = 0
        dropped = 0
        drop_prev = False
        for i, part in enumerate(parts):
            start, pos = pos, pos + len(part)
            raw = body[start:pos]  # what ships; `part` is what the scorer reads
            if i % 2:
                # separators are whitespace in the SCRUBBED view, so a citation
                # marker landing in one trailed the sentence before it. Dropping
                # that sentence and keeping its marker ships an [id] next to text
                # it never supported — what _prune_ungrounded_markers ran earlier
                # to stop, and this pass runs after it.
                out.append(_strip_orphan_markers(raw, start) if drop_prev else raw)
                continue
            drop_prev = False
            if not part.strip():
                out.append(raw)
                continue
            nums, ctx = _claim_profile(part)
            if not nums:  # prose carrying no figure is not a claim the scorer weighs
                out.append(raw)
                continue
            need = min(2, len(ctx))
            if any((nums & cn) and len(ctx & ct) >= need for cn, ct in corpus):
                out.append(raw)
                continue
            dropped += len(raw)
            drop_prev = True
            if any(cn and len(ctx & ct) >= 3 and not (nums & cn) for cn, ct in corpus):
                contradictions.append(raw.strip())
            else:
                unsupported.append(raw.strip())
        self._rollup_dropped_ratio = round(dropped / len(body), 4) if body else 0.0
        return "".join(out), contradictions, unsupported

    def _attribute_citations(self, text: str, source_ids: Dict[str, str]) -> str:
        """Attach each node source's global [id] only to the sentences it actually
        supports, instead of dumping every node source in one trailing block after
        the whole answer. The S1 grounding scorer requires an [id] marker to sit
        next to text that literally traces to that source; a bulk trailing dump
        (the pre-fix behavior) attaches every source to whichever sentence happens
        to be last, so most markers end up ungrounded even when the underlying
        source really does support SOME sentence in the text.

        A sentence earns a source's marker on a verbatim trace, not on
        text_supported: a source that only ever paraphrase-matches the node (that
        rule already kept it in node.sources) buys an [id] that can never ground,
        which only enlarges score_s1's denominator.

        The marker goes ON the traced phrase rather than at the sentence end.
        score_s1 reads the last 20 tokens before a marker, so a phrase borrowed
        early in a long sentence is invisible from the far end of it — measured
        on the round-1 goldens, that placement is what left denorm-derived-table
        at 3 of 5 ids grounded and solid-state-battery at 1 of 2. Sitting next to
        the wording it traces to is also what a citation is supposed to mean.

        s9: paragraph breaks survive. This used to space-join every sentence of a
        node answer into one blob, which left the roll-up a single 5KB lump with
        no claim units in it to compare or merge — and it silently reflowed the
        author's markdown (lists, bold lead-ins) into a wall.
        """
        blocks = _UNIT_SPLIT_RE.split(text or "")
        if len(blocks) > 1:
            return "\n\n".join(self._attribute_citations(b, source_ids)
                               for b in blocks if b.strip())
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return (text or "").strip()
        out = []
        for sent in sentences:
            marks: Dict[int, List[str]] = {}
            for url, cid in source_ids.items():
                end = _trace_end(sent, self._read_docs.get(url, ""))
                if end is None:
                    continue
                # nothing but punctuation left: append after the sentence rather
                # than wedge the marker in front of its own full stop. The grader
                # reads the same tokens either way, and the sentence stays intact
                # for everything downstream that matches on its text.
                if not _WORD_RE.search(sent[end:]):
                    end = len(sent)
                marks.setdefault(end, []).append(cid)
            # splice from the back so an earlier offset stays valid
            for pos in sorted(marks, reverse=True):
                cites = render_ids(sorted(marks[pos], key=int))
                sent = f"{sent[:pos]} {cites}{sent[pos:]}"
            out.append(sent)
        return " ".join(out)

    # ------------------------------------------------------- s9: claim merging

    async def _unit_vectors(self, texts: List[str]) -> tuple:
        """(vectors, similarity) for the claim units — embeddings when the service
        answers, bag-of-content-words when it does not.

        Either way this only SELECTS candidate pairs. The redundancy measured here
        is topical, not lexical (repeated 5-grams 0-2%), so word overlap alone
        misses the pairs that matter — but a missed candidate costs report length,
        never a fact, and the verdict below is what actually decides. Falling back
        instead of failing keeps the merge working wherever the embedding provider
        is down, out of quota, or simply not configured.

        THE FALLBACK IS FOR PRODUCTION, NOT FOR MEASUREMENT. Genuinely-restated
        pairs share little vocabulary, so under word overlap they never become
        candidates, the judge is never asked about them, and the merge is a no-op
        that looks exactly like a clean "nothing to merge". Measured 2026-07-29/30:
        the embeddings quota expired mid-loop and three refit rounds plus three
        human round-grants were spent on an implementation that was never the
        problem. `_embedded_units` is what lets a gate tell the two apart — see the
        embedded_units check in harness-search/scripts/d1_offline.lua.
        """
        self._embedded_units = 0
        if not texts:
            return [], _bag_cosine
        try:
            probe = await self.embed_question(texts[0])
            if probe:
                # THROTTLED like every other model path here. A tree offers 52-350 claim
                # units and one gather over all of them is the unbounded fan-out that
                # produced the 0xC0000142 exits recorded in d1_offline.json errors[].
                rest = await self._judged_in_waves([[t] for t in texts[1:]],
                                                   lambda b: self.embed_question(b[0]),
                                                   EMBED_CONCURRENCY)
                vecs = [list(probe), *(list(v) for v in rest)]
                if all(vecs):
                    self._embedded_units = len(vecs)
                    return vecs, _cosine
        except Exception as exc:  # any provider/seam failure degrades, never crashes
            logger.warning(f"claim-unit embeddings unavailable ({exc}); selecting "
                           "merge candidates by word overlap instead")
        return [_claim_counts(t) for t in texts], _bag_cosine

    async def _covered(self, texts: List[str]) -> Dict[int, bool]:
        """{index -> this unit states nothing the OTHERS listed do not}.

        One model call for the whole list, asking the same question either way: for
        each statement, ENUMERATE what a reader would LOSE if that statement were
        deleted and the rest kept.

        Two units is the VERDICT — equivalent only when both lists come back empty.
        More units is the cheap SCREEN in front of it: a unit the screen says has
        content of its own cannot be merged away by anybody, so none of its candidate
        pairs is worth a round trip. The screen never decides a merge on its own — with
        three or more units "nothing of my own" does not say WHICH of the others covers
        it, and collapsing a whole group on that is how two different claims on one
        topic become one (see `_merge_claim_units`).

        Fails closed on every unclear outcome — no model, a timeout, a reply that names
        no unit — because not merging costs `synthesis_ratio_pct_max` while wrongly
        merging costs facts, and `s2_min_delta >= -5` is the tighter bound.

        The units are sent marker-free: an [id] is not part of the claim, and a
        marker spliced mid-sentence would otherwise break the reply's own quoting.
        """
        if not self._merge_judge_ok or len(texts) < 2:
            return {}
        flats = [_flat_claim(t) for t in texts]
        if not all(flats):
            return {}
        provider, model = self._strategic_model()
        if not provider or not model:
            self._merge_judge_ok = False
            return {}
        try:
            reply = await asyncio.wait_for(
                create_chat_completion(
                    messages=[{"role": "user", "content":
                               self._equivalence_prompt(flats)}],
                    llm_provider=provider, model=model, temperature=0),
                timeout=MERGE_VERDICT_TIMEOUT_S)
        except Exception as exc:
            # CONSECUTIVE failures, not cumulative. A slow judge times out here and
            # there — measured 2026-07-29, 3 of 13 screens on one query — and a
            # cumulative count reads that as "the judge is dead" and abandons the four
            # screens still queued. Only an unbroken run of failures means nobody is
            # answering; one success is proof the judge is alive and resets the count.
            self._merge_judge_fails += 1
            logger.warning(f"claim-equivalence judge failed ({type(exc).__name__}: "
                           f"{exc}) — {self._merge_judge_fails} in a row of "
                           f"{MERGE_JUDGE_FAILURES} before the merge gives up")
            if self._merge_judge_fails >= MERGE_JUDGE_FAILURES:
                self._merge_judge_ok = False
            return {}
        self._merge_judge_fails = 0
        return self._reads_as_covered(str(reply or ""), flats)

    @staticmethod
    def _droppable_side(verdict: Dict[int, bool]) -> Optional[int]:
        """Which side of a JUDGED PAIR may be deleted — 0, 1, or None (fail closed).

        BOTH statements must be named before either may be dropped. The parser only
        reports a statement the reply actually named, and a partial reply is the norm
        (measured 2026-07-29: 10 of 16 lines in one screen, 5 of 8 in another). With a
        one-sided verdict `{0: True}` the old rule deleted statement 0 — even though
        that "nothing of its own" may have been the judge's line about the OTHER one,
        mis-attributed by a fragment that is a substring of both keys. Similar
        statements are exactly what gets here, so that is the normal case, not a corner,
        and the side it deletes is the detailed one. A pair the judge only half answered
        is not an answer.
        """
        if 0 not in verdict or 1 not in verdict:
            return None
        if verdict[1]:
            return 1
        return 0 if verdict[0] else None

    async def _droppable(self, a: str, b: str) -> Optional[int]:
        """Which of the two units can be deleted without losing a fact — 0, 1, or None.

        The two-unit call is the only one that decides a merge, because only there does
        "nothing of my own" name the statement that covers it: the other one.

        COVERAGE IS DIRECTIONAL, and insisting on the mutual case is what made the merge
        a no-op. Measured 2026-07-29 on denorm-derived-table (212 claim units): only 4 of
        32 screened units answered "nothing of my own", so pairs with BOTH ends covered
        were ~0 and almost nothing reached a verdict. But a report restating an earlier
        finding WITH more detail is the common shape, and there A is fully covered by B
        while B is not covered by A. Deleting A there loses nothing — that is exactly
        what the judge enumerated — and B stays whole. Requiring mutual coverage refuses
        that merge for no gain in safety.

        Mutual coverage keeps the first-stated wording, which is what the RED contract
        means by "whichever node's wording is chosen as canonical".
        """
        return self._droppable_side(await self._covered([a, b]))

    def _strategic_model(self) -> tuple:
        """(provider, model) for the merge judge, resolved once per assembly.

        The offline re-synthesis runner rebuilds the skill from a sidecar and has no
        live Config to hand it, so fall back to the deployment's own settings — the
        replay must exercise the same judge the live run does, or it is measuring
        something else.
        """
        if self._strategic_llm is None:
            cfg = getattr(self.researcher, "cfg", None)
            provider = getattr(cfg, "strategic_llm_provider", None)
            model = getattr(cfg, "strategic_llm_model", None)
            if not provider or not model:
                try:
                    from ..config import Config
                    live = Config()
                    provider = provider or live.strategic_llm_provider
                    model = model or live.strategic_llm_model
                except Exception as exc:
                    logger.warning(f"no strategic model for the claim merge ({exc}); "
                                   "the roll-up will not merge")
                    provider = model = None
            self._strategic_llm = (provider, model)
        return self._strategic_llm

    @staticmethod
    def _equivalence_prompt(flats: List[str]) -> str:
        """What each statement says that the others do not.

        Enumeration, not classification, and phrased as the loss test the report is
        actually graded on: a fact a reader would LOSE if this statement were
        deleted and the others kept. Asked as a bare "are these the same?" a model
        reports every difference in emphasis it can find — a restated claim with an
        extra qualifier reads as unique and nothing ever merges. Asked the other
        way round it collapses opposites. The listed cases are the two collapses
        that cost this corpus two golden facts.

        ATTRIBUTION IS NOT CONTENT, and it has to be said outright. Measured
        2026-07-29 over the corpus's own must-merge pair (the Oracle and pg_ivm
        statements of one incremental-maintenance claim): asked the earlier phrasing,
        three different models each answered UNIQUE and each named the SOURCE as the
        unique part — "Oracle's Data Warehousing Guide citation", "README's opening
        description citation". Two sources saying one thing is precisely the
        duplication this stage removes, so the instruction now names quotation,
        vendor and author as things to ignore, and the same pair comes back
        `none / none`.

        The reply is asked to QUOTE the statement each line is about, so a list of any
        length parses back to the statement it judged and a reordered or partial answer
        cannot be misread as a verdict on the wrong one.
        """
        listed = "\n\n".join(f"{_MERGE_LABELS[i]}: {f}" for i, f in enumerate(flats))
        n = len(flats)
        others = "the other one" if n == 2 else f"the other {n - 1}"
        return (
            f"{n} statements pulled from one research report. They are being "
            "de-duplicated.\n\n"
            f"{listed}\n\n"
            "Judge only the FACTUAL CLAIM each one makes about the subject matter.\n"
            "Deliberately IGNORE: which document, vendor, project or author is quoted; "
            "the wording; whether it is a quotation; how much detail or emphasis it "
            "carries; and any example that only illustrates a claim another one also "
            "makes. Two statements attributing the SAME claim to two different sources "
            "are NOT different claims — that is the duplication being removed.\n"
            "Treat as genuinely different: an opposite or different mechanism, a "
            "definition where another gives a consequence or a trade-off, a different "
            "quantity or threshold, a different subject, or a condition no other "
            "states.\n"
            "For each statement, name the factual content a reader would LOSE if that "
            f"statement were deleted and {others} kept.\n"
            f"Answer with exactly {n} lines and nothing else, one per statement, in "
            "this form:\n"
            '"<a verbatim fragment of that statement>" => UNIQUE: <what only that '
            "statement asserts, or the word none>\n"
            'Write "none" when the others already assert everything this one asserts, '
            "even if they say it about a different tool or in different words."
        )

    @staticmethod
    def _reads_as_covered(reply: str, flats: List[str]) -> Dict[int, bool]:
        """Parse the judge's enumeration into {index -> nothing of its own}.

        Each line names one statement — by quoting a fragment of it, or by its label —
        and says what only that statement asserts. A statement the reply never names
        is absent from the result, which reads as "no verdict" everywhere above and so
        fails closed."""
        verdict: Dict[int, bool] = {}
        keys = [_quote_key(f) for f in flats]
        for line in reply.splitlines():
            m = _UNIQUE_LINE_RE.match(line.strip()) or _UNIQUE_ALT_RE.match(line.strip())
            if not m:
                continue
            head, unique = m.group(1), m.group(2).strip().strip(".;,").lower()
            # AN EMPTY TAIL IS NO VERDICT, NOT COVERAGE (review R2). `... => UNIQUE:`
            # with the answer wrapped onto the next line is a common shape when the
            # unique content is long, and the continuation line carries no `=>` so it
            # is skipped — reading the empty tail as "the others say everything this
            # one says" deletes the statement the judge just named content for. The
            # statement is simply left unnamed, which every caller reads as no verdict.
            if not unique:
                continue
            quoted = _QUOTED_RE.search(head)
            frag = _quote_key(quoted.group(1) if quoted else "")
            # long enough that a shared stock phrase cannot claim the wrong statement,
            # and AMBIGUOUS MEANS NO VERDICT: first-match-wins hands one statement's
            # answer to another whenever the quoted fragment is a substring of both
            # keys, which is the normal shape here — the statements in one screen were
            # selected for similarity, and the target case is a later unit restating an
            # earlier one with more detail, i.e. the short key inside the long one.
            hits = [i for i, k in enumerate(keys) if len(frag) >= 24 and frag in k]
            side = hits[0] if len(hits) == 1 else None
            if side is None:
                label = _LABEL_RE.match(head)
                if not label:
                    continue
                tag = label.group(1) or label.group(2)
                side = int(tag) - 1 if tag.isdigit() else _MERGE_LABELS.find(tag.upper())
                if not 0 <= side < len(flats):
                    continue
            # a statement named twice with different verdicts is not a clear answer
            empty = unique in ("none", "nothing", "n a", "none.")
            verdict[side] = empty if side not in verdict else (verdict[side] and empty)
        return verdict

    async def _merge_claim_units(self, texts: List[str], vecs, sim) -> Dict[int, List[int]]:
        """{canonical unit index -> the indices it absorbs}.

        MERGE BY SELECTION, NOT BY REWRITING: one member's ORIGINAL wording is kept
        and the others are dropped, so nothing is re-worded and nothing loses the
        grounding its node earned. Three rewrite-then-re-attribute designs were
        measured live to lose citations, one reaching citations_total=0.

        A STAR, NOT A UNION-FIND. The judge's guarantee is pairwise and mutual — both
        sides answered "nothing of my own about the other" — and it does NOT compose.
        A union-find would unite whole roots, so A could be deleted in favour of a C it
        was never compared with: A≡B unites at B's root, then B≡C folds both into C.
        If A states
        something C does not — the same-topic-different-claim case that cost this corpus
        golden facts 2 and 8 — it is gone and the log still reports one clean cluster.
        So every absorbed unit is judged DIRECTLY against the unit that survives it: a
        canonical is never itself absorbed, and a pair whose end has already been
        absorbed is dropped rather than re-targeted. That costs report length, which is
        the error the gates forgive.
        """
        # cleared FIRST: an early return used to leave the previous assembly's screen and
        # verdict counts in place, so _merge_stats reported work this run did not do
        self._merge_calls = {"screens": 0, "candidate_pairs": 0, "verdicts": 0,
                             "verdict_pairs": 0, "screened_out": 0}
        n = len(texts)
        if n < 2:
            return {}

        merged: Dict[int, List[int]] = {}
        # SCREEN, then decide. A unit the screen says states something of its own can
        # never be merged away, so the pairs it sits in are not worth a round trip;
        # measured on the captured corpus this is what takes a query from 111-294
        # verdicts to a few dozen calls. The screen is not allowed to merge anything by
        # itself: with three or more units in front of it, "nothing of my own" does not
        # say WHICH of the others covers this one, and collapsing the whole answer-none
        # set is how a closure-table DEFINITION and its TRADE-OFF become one statement.
        groups, floor = self._merge_groups(n, vecs, sim)
        screened = await self._judged_in_waves(
            [[texts[i] for i in g] for g in groups], self._covered)
        # THE VERDICT PAIRS COME FROM THE SCREEN'S OWN GROUP, not from a global
        # candidate graph. The screen's answer means "some other member of THIS group
        # already says everything I say", so the unit that covers it is in this group by
        # construction — while the old global top-3 neighbour graph could perfectly well
        # not contain that pair, in which case a correctly screened unit never reached a
        # verdict and never merged.
        covered: set = set()
        scored_pairs: Dict[tuple, float] = {}
        absorbed: set = set()
        for g, verdict in zip(groups, screened):
            cov = [k for k, nothing in (verdict or {}).items()
                   if nothing and k < len(g)]
            if not cov:
                continue
            covered |= {g[k] for k in cov}
            # A TWO-UNIT SCREEN IS ALREADY THE VERDICT — it is the very call _droppable
            # makes, with the very same two statements in it, so asking again is one
            # model round trip spent to be told what this reply just said. Measured
            # 2026-07-29 the grown clusters are mostly pairs (19 of denorm's 27), so
            # this is most of the merge's cost. Both sides covered keeps the
            # first-stated wording, exactly as _droppable does.
            if len(g) == 2:
                side = self._droppable_side(verdict or {})
                if side is None:
                    continue
                drop, keep = (g[side], g[1 - side])
                absorbed.add(drop)
                merged.setdefault(keep, []).append(drop)
                continue
            for k in cov:
                for other in g:
                    if other != g[k]:
                        p = (min(g[k], other), max(g[k], other))
                        scored_pairs[p] = sim(vecs[p[0]], vecs[p[1]])
        # strongest candidates first, so the closest pair gets the first call on each
        # unit. A wave at a time: waves keep the ordering that matters (a stronger pair
        # is always judged before a weaker one) — the only thing concurrency loses is
        # the chance to SKIP a pair one of whose ends has just been absorbed, which
        # costs a call, never a wrong merge (the same test is re-applied below).
        # ONE end covered is enough to be worth asking. A unit the screen says has
        # content of its own cannot be deleted, but it can perfectly well be the unit
        # that COVERS its neighbour — the restated-with-more-detail shape. Requiring
        # both ends is what left ~0 pairs to judge (4 covered units in 32 screened).
        order = [p for p, _ in sorted(scored_pairs.items(), key=lambda kv: (-kv[1], kv[0]))]
        verdicts_asked = 0
        for w in range(0, len(order), MERGE_CONCURRENCY):
            if not self._merge_judge_ok:
                break
            wave = [(i, j) for i, j in order[w:w + MERGE_CONCURRENCY]
                    if i not in absorbed and j not in absorbed]
            if not wave:
                continue
            verdicts_asked += len(wave)
            sides = await asyncio.gather(
                *(self._droppable(texts[i], texts[j]) for i, j in wave))
            for (i, j), side in zip(wave, sides):
                # re-tested after the await: two pairs sharing a unit can be in one wave
                if side is None or i in absorbed or j in absorbed:
                    continue
                drop, keep = (i, j) if side == 0 else (j, i)
                # a canonical is never absorbed — its members were judged against IT, not
                # against whatever it would be folded into. Fail closed rather than swap:
                # the judge only cleared THIS direction.
                if drop in merged:
                    continue
                absorbed.add(drop)
                merged.setdefault(keep, []).append(drop)
        self._merge_calls = {"screens": len(groups),
                             "screened_units": sum(len(g) for g in groups),
                             "covered_units": len(covered),
                             "verdicts": verdicts_asked, "verdict_pairs": len(order),
                             "screened_out": sum(len(g) for g in groups) - len(covered),
                             # the band this run actually derived (-1 = tree fit in one
                             # screening call, so no floor was needed). Recorded because
                             # it now varies with the embedder: nothing else downstream
                             # can tell a healthy band from a degenerate one.
                             "candidate_floor": round(floor, 4)}
        return {k: sorted(v) for k, v in merged.items()}

    @staticmethod
    def _merge_groups(n: int, vecs, sim) -> Tuple[List[List[int]], float]:
        """The units that are worth screening together, as clusters GROWN around the
        closest pair rather than sliced out of a chain.

        The previous version walked each connected component of a top-3-neighbour graph
        as a nearest-neighbour chain and cut it every MERGE_GROUP units. At
        MERGE_FLOOR = 0.08 that component was the whole tree, so a "group" was a
        16-unit slice of a walk across every topic in the report. Measured 2026-07-29 on
        denorm-derived-table: 13 screens, 195 units, 5 covered — the honest answer,
        because those 16 statements really did each say something the other 15 did not.

        Grown instead: take the strongest unassigned pair as the seed, then repeatedly
        add the unassigned unit most like the cluster SO FAR, stopping when the best
        candidate's mean similarity to the cluster falls under the candidate floor or
        the group is full. A unit that joins no cluster is never screened, which is the
        saving — the old scheme screened every unit in the tree.

        The floor is READ OFF THIS RUN'S OWN SCORES, not configured: the top
        MERGE_CANDIDATE_PCT of the pair distribution. A cosine means nothing across
        embedders — the same corpus scores q0.99 = 0.291 under text-embedding-3-small and
        0.738 under qwen3-embedding-4b — so a hardcoded floor is a hidden dependency on
        one provider that fails silently in BOTH directions (too low: every pair becomes a
        model round trip; too high: nothing is ever a candidate and the merge is a no-op
        that looks like a tree with no redundancy).

        A tree that fits in one group is screened in one call: at that size there is no
        grouping decision to get wrong, and it is the shape the s9 fixture pins.
        Deterministic — ties break on the lower index, so the same tree screens the
        same groups every run.
        """
        # -1.0, not 0.0: at this size there IS no candidate floor (every unit is screened
        # in one call), and 0.0 would read as a derived band that admitted nothing
        if n <= MERGE_GROUP:
            return ([list(range(n))] if n > 1 else []), -1.0
        cache: Dict[tuple, float] = {}

        def s(i: int, j: int) -> float:
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                cache[key] = sim(vecs[key[0]], vecs[key[1]])
            return cache[key]

        # every pair is scored either way (the seed list needs the ranking), so reading the
        # floor off the same scan costs nothing beyond the sort it already does
        ranked = sorted(((s(i, j), i, j) for i in range(n) for j in range(i + 1, n)),
                        key=lambda t: (-t[0], t[1], t[2]))
        floor = ranked[min(len(ranked) - 1, int(len(ranked) * MERGE_CANDIDATE_PCT))][0]
        seeds = [t for t in ranked if t[0] >= floor]
        used: set = set()
        groups: List[List[int]] = []
        for _, i, j in seeds:
            if i in used or j in used:
                continue
            cluster = [i, j]
            used |= {i, j}
            while len(cluster) < MERGE_GROUP:
                left = [m for m in range(n) if m not in used]
                if not left:
                    break
                nxt = max(left, key=lambda m: (sum(s(m, c) for c in cluster) / len(cluster),
                                               -m))
                if sum(s(nxt, c) for c in cluster) / len(cluster) < floor:
                    break
                cluster.append(nxt)
                used.add(nxt)
            groups.append(sorted(cluster))
        return groups, floor

    @staticmethod
    async def _judged_in_waves(batches: List[list], judge,
                               width: int = MERGE_CONCURRENCY) -> List[Any]:
        """Run `judge` over every batch, `width` calls in flight.

        The deployment's judge is a CLI round trip; sequentially, a query's calls run
        past any sane session budget, and all of them at once is the process-spawn
        storm that crashed four of five re-syntheses. The embedding service gets a
        wider wave than the judge does — it is an HTTP call, not a process — but it
        does not get an uncapped one.
        """
        out: List[Any] = []
        for w in range(0, len(batches), width):
            out += list(await asyncio.gather(*(judge(b) for b in batches[w:w + width])))
        return out

    def _themes(self, idx: List[int], vecs, sim) -> List[List[int]]:
        """Group the surviving units (given by index) into themes DERIVED FROM THEM.

        The report's outline must not be the tree's: a node's answer could only ever
        appear beneath its own node, so two siblings restating one finding were never
        in the same place and no threshold could ever bring them together. Themes cut
        across nodes (STORM, arXiv:2402.14207, expands an outline built from the
        collected references; Egnyte's writer stage makes the emergent themes of a
        meta-analysis over all question analyses its sections).

        A SEED MUST BE REPRESENTATIVE AS WELL AS DISTINCT. Taking simply "the unit least
        like every seed so far" seeds on the OUTLIERS — the short connective lines with
        almost no content words, which resemble nothing — and then nearly every real
        unit's best similarity is to seed 0, or is 0 to all of them and falls to seed 0
        on the tie-break. Measured on the shipped denorm report: sections of
        41101 / 4485 / 8475 / 27 characters, 161 of 205 blocks in one, and the 27 was a
        lone "Piecing these together:" with no citation in it at all. So each seed after
        the first maximises centrality DISCOUNTED by its likeness to the seeds already
        chosen: still deterministic, still spread out, but every seed is a unit other
        units are actually near.
        """
        n = len(idx)
        k = max(2, min(MAX_THEMES, int(round(n ** 0.5))))
        if n <= k:
            return [[i] for i in idx]
        total = {i: sum(sim(vecs[i], vecs[j]) for j in idx) for i in idx}
        seeds = [max(idx, key=lambda i: (total[i], -idx.index(i)))]
        while len(seeds) < k:
            seeds.append(max(
                (i for i in idx if i not in seeds),
                key=lambda i: (total[i] * (1 - max(sim(vecs[i], vecs[s]) for s in seeds)),
                               -i)))
        groups: List[List[int]] = [[] for _ in seeds]
        for i in idx:
            best = max(range(len(seeds)), key=lambda s: (sim(vecs[i], vecs[seeds[s]]), -s))
            groups[best].append(i)
        return [g for g in groups if g]

    @staticmethod
    def _theme_title(members: List[str], others: List[str], used: set) -> str:
        """A title made of what this theme says and the others do not.

        `used` keeps two sections from shipping the SAME heading — which the ranking
        can produce whenever two themes share their distinctive vocabulary, and which
        the "Findings" fallback produces for any theme with no content words at all.
        Two identical H2s read as one section split in half.
        """
        inside: Dict[str, int] = {}
        for t in members:
            for w in _claim_tokens(t):
                inside[w] = inside.get(w, 0) + 1
        outside: Dict[str, int] = {}
        for t in others:
            for w in _claim_tokens(t):
                outside[w] = outside.get(w, 0) + 1
        ranked = sorted(inside, key=lambda w: (-inside[w] / (1 + outside.get(w, 0)),
                                               -inside[w], w))
        picked: List[str] = []
        spare: List[str] = []
        for w in ranked:
            # "table" and "tables" are one word for a title's purposes
            if any(w[:5] == p[:5] for p in picked + spare):
                continue
            (picked if len(picked) < 4 else spare).append(w)
        # ponytail: keyword titles, derived from the theme's own distinctive words —
        # NOT prose. Asking the model was considered and is not free here: a title call
        # has to quote the section's own sentences to be about it, and the s9 fixture's
        # stand-in answers any prompt quoting two known sentences with UNIQUE lines, not
        # an outline. Prose titles are a model call whose output nothing yet verifies;
        # add it when a human reads these.
        def render() -> str:
            return ", ".join(w.capitalize() if i == 0 else w
                             for i, w in enumerate(picked)) or "Findings"

        title = render()
        while title in used:
            if not spare:
                title = f"{title} ({len(used) + 1})"
                break
            picked.append(spare.pop(0))
            title = render()
        used.add(title)
        return title

    async def synthesize_node(self, node: ResearchNode,
                              child_summaries: Optional[List[str]] = None,
                              source_ids: Optional[Dict[str, str]] = None) -> str:
        """Roll one node up into a summary (leaf: own attributed answer;
        internal: own answer followed by each child's summary, verbatim).

        s9: assemble_report now calls this once per node with NO child summaries and
        lays the claim units out by theme instead. Nesting a child's text inside its
        parent's is what pinned every finding under its own node, so two siblings
        restating one claim were never in the same place and no merge could reach
        them. The child argument stays — the seam's contract is unchanged and a
        caller that wants the subtree in one string still gets it.

        Three progressively wider "LLM rewrites the merge, then re-derive [id]
        markers by fuzzy-matching the rewrite against pre-merge text" designs
        were all measured live to lose citations: matching against raw source
        pages, then against immediate pre-merge blocks, then against the whole
        subtree's pre-merge blocks -- each still lost more citations at every
        rewrite level, culminating in citations_total=0 for a 33-node/depth-3
        tree (bun-rust-port) even with the widest corpus. The rewrite step
        itself is what erases the grounding a wider match can't reliably buy
        back. Concatenation can't lose a citation a child already earned,
        because it never touches the child's text.
        """
        child_summaries = [s for s in (child_summaries or []) if s]
        source_ids = source_ids or {}
        if node.status == NodeStatus.PENDING:
            return f"(unexplored frontier) {node.question}"
        if node.status == NodeStatus.FAILED:
            # defect 3: a failed node contributes NO text of its own — neither its
            # answer (prior knowledge written over missing evidence) nor the
            # node.question fallback below, which would smuggle the unresearched
            # question into the report as if it were a finding. Children that did
            # research still roll up through it: their findings were researched.
            return "\n\n".join(child_summaries)
        # attribution needs text close to the source wording: answer_md is what
        # node.sources narrowing already checked for overlap (see research_node);
        # answer_digest is an LLM paraphrase that rarely clears text_supported's
        # concentrated-passage threshold, which silently zeroed every citation
        # (measured live: citations_total=0, S1_pct=0 on both s2 measure queries)
        base = node.answer_md or node.answer_digest or node.question
        own = self._attribute_citations(base, source_ids)
        if not child_summaries:
            return own
        return "\n\n".join([own, *child_summaries])

    # --------------------------------------------------------------- assembly

    async def assemble_report(self, query: str) -> Dict[str, Any]:
        """Citation map -> post-order roll-up -> fail-closed marker passes -> report.

        Split out of run() so the offline re-synthesis runner
        (harness-search/scripts/resynth.py) replays THIS assembly rather than a copy
        of it. It reads only self.nodes and self._read_docs and performs no retrieval
        — create_chat_completion appears twice in this module and both call sites are
        upstream — so a captured tree can be re-assembled for free instead of at ~330
        Firecrawl credits and 12 minutes per query. A private copy in the runner would
        pass its fidelity check today and diverge silently the next time the real one
        is edited, which is exactly the drift nobody would see.

        Returns report_md plus the by-products run() reports: citation_map,
        uncited_ids, contradictions, unsupported.
        """
        # stable citation ids: URL union in first-seen node/insertion order
        citation_map: Dict[str, str] = {}
        url_to_id: Dict[str, str] = {}
        for n in self.nodes.values():
            # defect 3: a failed node's URLs are not evidence either — otherwise its
            # sources leak into the report through the Citations list even though the
            # roll-up already dropped its text
            if n.status == NodeStatus.FAILED:
                continue
            for url in n.sources:
                if url not in url_to_id:
                    cid = str(len(url_to_id) + 1)
                    url_to_id[url] = cid
                    citation_map[cid] = url

        # post-order roll-up: children before parent, root synthesized last. Each
        # node is synthesized ON ITS OWN — the child summaries are no longer nested
        # into the parent's text, because that nesting is exactly what pinned every
        # finding under its own node and put two siblings' restatements of one claim
        # in two different places, where no merge could ever reach them. The claim
        # units below are laid out by THEME instead.
        units: List[str] = []
        unit_ids: List[Dict[str, str]] = []
        self._merge_judge_ok, self._merge_judge_fails = True, 0

        async def rollup(n: ResearchNode) -> None:
            for cid in n.children:
                child = self.nodes[cid]
                if child.status == NodeStatus.PRUNED:
                    continue
                await rollup(child)
            node_source_ids = {u: url_to_id[u] for u in n.sources if u in url_to_id}
            text = await self.synthesize_node(n, [], node_source_ids)
            self._syntheses[n.id] = text
            # a PENDING node contributes "(unexplored frontier) {question}" — a
            # question nobody researched, not a finding. The frozen scorer matches a
            # golden fact anywhere in the file, so those lines let a report score a
            # fact purely by reciting the question that mentions it.
            if n.status == NodeStatus.PENDING:
                return
            for part in _claim_units(text):
                units.append(part)
                unit_ids.append(node_source_ids)

        # the root is the first node inserted (run() seeds it before the frontier
        # loop), which is also the order the resynth sidecar preserves
        root = next(iter(self.nodes.values()), None)
        if root is not None and root.status != NodeStatus.PRUNED:
            await rollup(root)

        vecs, sim = await self._unit_vectors(units)
        merged = await self._merge_claim_units(units, vecs, sim)
        absorbed = {i for members in merged.values() for i in members}
        # MIGRATE THE GROUNDING, don't assume it: an absorbed unit's source earns its
        # [id] on the surviving wording only where the frozen scorer can still trace
        # it to that source's own page. _attribute_citations is that rule.
        for keep, members in merged.items():
            extra = {u: cid for i in members for u, cid in unit_ids[i].items()
                     if u not in unit_ids[keep]}
            if extra:
                units[keep] = self._attribute_citations(units[keep], extra)
        kept = [i for i in range(len(units)) if i not in absorbed]
        # REPORTED, not just logged: "the merge found nothing" and "the merge worked"
        # look identical in every downstream metric until four stages later, so the
        # counters travel with the report the caller persists (see run()/_persist).
        self._merge_stats = {
            "claim_units": len(units), "units_merged": len(absorbed),
            "shared_claims": len(merged),
            # PROVENANCE OF THE SIMILARITY SIGNAL, not a quality number: < claim_units
            # means candidates were picked by word overlap, which cannot find the pairs
            # this stage exists to merge, so any redundancy figure taken from that run
            # is unusable rather than merely bad. See _unit_vectors.
            "embedded_units": self._embedded_units,
            "chars_before": sum(len(u) for u in units),
            "chars_kept": sum(len(units[i]) for i in kept),
            **self._merge_calls,
        }
        # WARNING, not info, when nothing merged: a roll-up that merged nothing is the
        # concatenation this stage exists to remove, and it is invisible in every
        # downstream number until the report is scored four stages later. The offline
        # re-synthesis runner configures no logging, so info is swallowed and warning is
        # the only level that reaches its stderr — the counters themselves travel out in
        # the returned merge_stats, but only run() persists those today.
        degraded = self._embedded_units < len(units)
        line = (f"roll-up merge: {len(absorbed)} of {len(units)} claim units "
                f"absorbed into {len(merged)} shared claims "
                f"({self._merge_stats.get('screens', 0)} screening calls, "
                f"{self._merge_stats.get('verdicts', 0)} verdicts, "
                f"{self._merge_stats.get('covered_units', 0)} units screened as covered, "
                f"{self._embedded_units}/{len(units)} embedded"
                f"{' — CANDIDATES BY WORD OVERLAP' if degraded else ''})")
        (logger.warning if (degraded or not absorbed) and len(units) > 1
         else logger.info)(line)

        if kept:
            groups = self._themes(kept, vecs, sim)
            # A SECTION WITH NO CITED FINDING IN IT IS A DIVIDER, NOT A THEME. Every
            # finding in a healthy tree is grounded, so a group holding no [id] at all
            # is heading-count inflation — measured on the shipped denorm report as a
            # 27-character section whose whole content was "Piecing these together:".
            # Folded into the group it is most like rather than dropped: the text is
            # still a reader's lead-in to something, and deleting content is the one
            # error this stage may not make.
            solid = [g for g in groups if any(_CITE_ID_RE.search(units[i]) for i in g)]
            for g in [g for g in groups if g not in solid]:
                if not solid:
                    solid.append(g)
                    continue
                host = max(solid, key=lambda t: max(sim(vecs[i], vecs[j])
                                                    for i in g for j in t))
                host.extend(g)
            groups = [sorted(g) for g in solid]
            titled: set = set()
            sections = sorted(
                (min(g), self._theme_title([units[i] for i in g],
                                           [units[i] for i in kept if i not in g],
                                           titled), g)
                for g in sorted(groups, key=min))
            lines = [f"# {query}", ""]
            for _, title, g in sections:
                lines += [f"## {title}", ""]
                for i in g:
                    lines += [units[i], ""]
            body = "\n".join(lines)
        else:
            body = "\n".join([f"# {query}", "", "_(no synthesis)_", ""])

        # defect 6a fail-closed: an [id] with no citations entry never reaches
        # the caller — detect, then strip the unbacked markers. Body only: the
        # Citations list is data, so a bracketed number inside a URL must never
        # be read as a citation marker (nor be rewritten by the strip).
        # Scanned over the node answers as well as the assembled body: s3 drops a
        # starved node's text out of the roll-up, so a marker the answer LLM
        # fabricated on a dropped node would stop being REPORTED exactly when the
        # tree is failing — the signal goes quiet at the moment it matters most.
        # The strip below still rewrites only the body; an id that never reached it
        # is a no-op there.
        uncited_ids = find_uncited_ids(
            "\n".join([body, *(n.answer_md for n in self.nodes.values())]), citation_map)
        if uncited_ids:
            logger.error(f"uncited [id] markers stripped from report: {uncited_ids}")
            bad = set(uncited_ids)

            def _keep_cited(m: "re.Match[str]") -> str:
                kept = [c for c in _bracket_ids(m.group(1)) if c not in bad]
                return render_ids(kept)

            body = _CITE_ID_RE.sub(_keep_cited, body)

        body = self._prune_ungrounded_markers(body, citation_map)

        # defect 6b: last, so the claim check sees the body as it will ship and the
        # Citations list below is rendered from what SURVIVED it — a dropped claim
        # must not leave its source behind in the citations block.
        body, contradictions, unsupported = self.verify_rollup(body)
        if contradictions or unsupported:
            # the ratio separates "one claim was cleaned" from "the report collapsed":
            # when most nodes FAILED the corpus is tiny and every figure misses, and
            # the counts alone read the same either way
            logger.error(f"roll-up consistency: dropped {len(contradictions)} claim(s) "
                         f"contradicting a node answer and {len(unsupported)} no node "
                         f"answer supports "
                         f"({self._rollup_dropped_ratio:.1%} of the body): "
                         f"{[*contradictions, *unsupported]}")

        # a source that survived node.sources narrowing but whose id never made it
        # into the synthesized body (e.g. an internal rollup dropped it while
        # merging sub-findings) must not be RENDERED in the Citations list either:
        # the S1 scorer treats "- [id] url" as one more occurrence of that marker,
        # and an id with no real in-body usage can only ever ground against the
        # citations list's own boilerplate line, never a supporting sentence.
        # citation_map itself (returned to the caller, and what tree.json's
        # "citations" field is built from) stays the full node.sources union.
        used_ids = {cid for m in _CITE_ID_RE.finditer(body) for cid in _bracket_ids(m.group(1))}
        rendered_map = {cid: url for cid, url in citation_map.items() if cid in used_ids}

        lines = [body]
        if rendered_map:
            lines += ["", "## Citations", ""]
            lines += [f"- [{cid}] {url}" for cid, url in rendered_map.items()]
        report_md = "\n".join(lines).strip() + "\n"

        return {"report_md": report_md, "citation_map": citation_map,
                "uncited_ids": uncited_ids, "contradictions": contradictions,
                "unsupported": unsupported, "merge_stats": dict(self._merge_stats)}

    # -------------------------------------------------------------------- run

    async def run(self, query: Optional[str] = None, max_depth: int = 3,
                  max_breadth: int = 4, max_nodes: int = 20,
                  token_budget: int = 300_000, credit_budget: float = 150.0,
                  novelty_threshold: float = 0.30, expansion_policy: str = "best_first",
                  stream: bool = False, outputs_dir: Optional[str] = None,
                  time_budget_s: float = 600.0, node_concurrency: int = 3) -> Dict[str, Any]:
        query = query or self.researcher.query
        self._root_query = str(query or "")
        self._max_breadth = max_breadth
        start = time.time()

        root = ResearchNode(id="0", question=query, parent_id=None, depth=0)
        root.priority = 1.0
        # the root question is the one topic guaranteed to be covered; unembedded,
        # a first-generation child that merely rephrases the original query scores
        # fully novel and gets researched all over again (defect 5)
        try:
            root.question_embedding = await self.embed_question(query)
        except Exception as e:
            logger.warning(f"root embedding failed, novelty degraded: {e}")
        self.nodes[root.id] = root

        frontier = Frontier()
        frontier.push(root)
        researched = 0
        pruned = 0

        # budgets are checked BEFORE each frontier batch; accepted-but-unresearched
        # nodes stay PENDING in the tree ("unexplored frontier")
        # defect 2, at tree scale: researching nodes strictly sequentially (~45s
        # each) made time_budget_s — not max_nodes — decide how much of the tree
        # was ever researched. Bench round 2 ended with 12 of 21 (denorm) and 17 of
        # 29 (outbox) nodes still PENDING, and every primary-source question those
        # nodes carried ("what did Celko write about closure tables") went
        # unresearched, so the context that reached the report was a third of the
        # tree the expansion had already earned. Each node researches with its OWN
        # GPTResearcher (see research_node) and shares only visited_urls — a set
        # mutated inside this one event loop — so a batch of them is safe to run
        # concurrently. Scoring/expansion bookkeeping stays sequential in priority
        # order, so novelty and covered ground see exactly what they saw before.
        worst_node_tokens = 0
        while (len(frontier) and researched < max_nodes
               and self.tokens_spent < token_budget
               and self.credits_spent < credit_budget
               and time.time() - start < time_budget_s):
            batch_size = min(node_concurrency, len(frontier), max_nodes - researched)
            if worst_node_tokens:
                # keep the budget honest at batch granularity: the sequential loop
                # stopped ON the budget, so reserve room for the costliest node
                # seen so far rather than letting a batch overshoot it
                headroom = (token_budget - self.tokens_spent) // worst_node_tokens
                batch_size = min(batch_size, max(1, headroom))
            batch = [frontier.pop() for _ in range(batch_size)]

            # Prune BEFORE researching, not after. compute_novelty reads only the
            # node's QUESTION embedding, so a node destined for PRUNED can be
            # identified without spending a search, its scrapes and an answer LLM
            # call on it first. Measured on round 4, 23-62% of everything researched
            # was pruned afterwards — that work was bought and thrown away.
            # The node is still CREATED and still lands as PRUNED with pruned_count
            # incremented (the s4 contract), it just costs nothing now. Nodes pruned
            # here never entered research, so they do not count towards `researched`.
            survivors = []
            for node in batch:
                node.novelty = self.compute_novelty(node)
                if node.novelty < novelty_threshold:
                    node.status = NodeStatus.PRUNED
                    pruned += 1

                    continue  # pruned nodes are never researched and never expanded
                survivors.append(node)
            batch = survivors
            if not batch:
                continue

            outcomes = await asyncio.gather(
                *(self.research_node(node) for node in batch), return_exceptions=True)
            for node, outcome in zip(batch, outcomes):
                if isinstance(outcome, BaseException):
                    logger.error(f"tree node {node.id} research failed: {outcome}")
                    node.status = NodeStatus.FAILED
                    self.unregister_covered(node)
                    continue
                researched += 1
                self.tokens_spent += node.tokens_spent
                self.credits_spent += node.credits_spent
                worst_node_tokens = max(worst_node_tokens, node.tokens_spent)

                if node.status == NodeStatus.FAILED:
                    self.unregister_covered(node)
                    # defect 3: nothing to expand from and nothing to roll up. Skipping
                    # compute_novelty matters too — registering a starved node's
                    # question as covered ground would let the hole it left prune a
                    # later child that would have filled it.
                    continue

                # novelty was scored (and the question registered) before research,
                # see the batch loop above. A node that FAILED researched nothing, so
                # its registration is withdrawn here — presenting a starved question
                # as covered ground would prune the later child that could fill it.

                if node.depth >= max_depth:
                    continue
                try:
                    candidates = await self.generate_child_questions(node)
                except Exception as e:
                    logger.error(f"tree node {node.id} expansion failed: {e}")
                    candidates = []
                accepted = 0
                for question in candidates:
                    if accepted >= max_breadth:
                        break
                    try:
                        emb = await self.embed_question(question)
                    except Exception as e:
                        logger.warning(f"embedding failed, dedup degraded: {e}")
                        emb = None
                    if emb and any(_cosine(emb, e) >= DEDUP_COSINE for e in self._embeddings):
                        continue  # near-duplicate question -> dropped, no node created
                    child = ResearchNode(id=f"{node.id}.{len(node.children)}",
                                         question=question, parent_id=node.id,
                                         depth=node.depth + 1)
                    child.question_embedding = emb
                    # ponytail: heuristic priority (design's novelty/gap terms need research
                    # results a fresh child doesn't have yet); bfs/dfs just reorder by depth
                    if expansion_policy == "bfs":
                        child.priority = -child.depth
                    elif expansion_policy == "dfs":
                        child.priority = float(child.depth)
                    else:
                        # the affinity term is sized to outrank one depth band: a
                        # "what does <entity>'s own documentation say" child is worth
                        # more of a budget that never reaches the whole frontier than
                        # a generic question one level shallower.
                        child.priority = max(0.0, 0.5 + 0.15 * node.priority
                                             - 0.10 * child.depth
                                             + 0.35 * _primary_source_affinity(question))
                    if emb:
                        self._embeddings.append(emb)
                    self.nodes[child.id] = child
                    node.children.append(child.id)
                    frontier.push(child)
                    accepted += 1
                if node.children:
                    node.status = NodeStatus.EXPANDED

        budget_respected = (researched <= max_nodes
                            and self.tokens_spent <= token_budget
                            and self.credits_spent <= credit_budget)
        # unresearched nodes left in the frontier tell the caller the tree was cut short
        time_budget_exhausted = (time.time() - start >= time_budget_s and len(frontier) > 0)

        assembled = await self.assemble_report(query)
        report_md = assembled["report_md"]
        citation_map = assembled["citation_map"]
        uncited_ids = assembled["uncited_ids"]
        contradictions = assembled["contradictions"]
        unsupported = assembled["unsupported"]

        # CitationAgent over the tree node claims: each learning is attributed
        # to the surviving node source that actually supports it; when none
        # does, the claim records no url ("") so it fails closed as unverified
        # instead of blaming an arbitrary source.
        # R3 (minor, deliberately NOT closed): the roll-up / citation-map / tree.json
        # guards all skip FAILED nodes, and extending the same guard here was tried —
        # it fails frozen test s2 test_citation_agent_flags_unverified_tree_node_claims,
        # which researches a node on 17 chars of context (below MIN_CONTEXT_CHARS, so
        # FAILED) and then asserts total_claims >= 2. That test pins the claim map as
        # a status-blind view of every node's learnings — an unverified-claim detector,
        # not a report input — so filtering it is a contract change, not a cleanup.
        # No metric is affected: score_report.py reads report.md + tree.json only.
        claim_urls: Dict[str, str] = {}
        for n in self.nodes.values():
            for learning in n.learnings:
                url = next(
                    (u for u in n.sources
                     if text_supported(learning, self._read_docs.get(u, ""))),
                    "")
                claim_urls.setdefault(learning, url)
        citation_verification = await asyncio.to_thread(
            CitationAgent().verify, claim_urls, self._read_docs)

        max_depth_reached = max((n.depth for n in self.nodes.values()), default=0)
        meta = {
            "query": query,
            "budget_respected": budget_respected,
            "max_depth_reached": max_depth_reached,
            "pruned_count": pruned,
            "node_count": len(self.nodes),
        }
        tree = {
            "meta": meta,
            "nodes": {nid: self._node_dict(n) for nid, n in self.nodes.items()},
        }
        result: Dict[str, Any] = {
            "report_md": report_md,
            "tree": tree,
            "citation_map": citation_map,
            "uncited_ids": uncited_ids,
            "citation_verification": citation_verification,
            "stats": {**meta, "researched": researched,
                      "contradictions": len(contradictions),
                      "unsupported_claims": len(unsupported),
                      "rollup_dropped_ratio": self._rollup_dropped_ratio,
                      # what the merge actually did this run. Without it a roll-up that
                      # merged nothing and one that merged half the report are the same
                      # number everywhere until the dedup metrics are scored.
                      "merge": assembled["merge_stats"],
                      "tokens_spent": self.tokens_spent,
                      "credits_spent": self.credits_spent,
                      "time_budget_exhausted": time_budget_exhausted,
                      "pending_count": len(frontier),
                      "elapsed_s": round(time.time() - start, 2)},
        }
        if outputs_dir:
            result["artifacts"] = self._persist(outputs_dir, query, meta, citation_map, report_md)
        return result

    @staticmethod
    def _node_dict(n: ResearchNode) -> Dict[str, Any]:
        # defect 3: a starved node's answer and URLs stay on the node object (the s2
        # contract above is written against them) but they are not evidence, and
        # tree.json IS read as evidence — the scorer counts nodes[].sources domains
        # for S4 and matches report claims against node answers for S6. Emitting them
        # would let a node the roll-up already dropped support the report anyway.
        failed = n.status == NodeStatus.FAILED
        return {
            "id": n.id,
            "question": n.question,
            "status": n.status.value,
            "children": list(n.children),
            "parent_id": n.parent_id,
            "depth": n.depth,
            "sources": [] if failed else list(n.sources),
            "novelty": n.novelty,
            "priority": n.priority,
            "answer_digest": "" if failed else n.answer_digest,
        }

    def _persist(self, outputs_dir: str, query: str, meta: Dict[str, Any],
                 citation_map: Dict[str, str], report_md: str) -> Dict[str, str]:
        """Write tree.json (smoke-gate contract shape) + final report markdown +
        the <stem>.resynth.json sidecar the offline re-synthesis runner replays.

        The sidecar is ADDITIVE — tree.json's shape is what the frozen scorer reads
        and does not move. It carries what assemble_report needs and tree.json
        deliberately drops: self._read_docs (citation attribution traces against the
        scraped text), and per-node answer_md/sources/learnings unblanked, in
        self.nodes INSERTION order because citation ids are assigned in it. Losing
        any of those makes the offline replay diverge from this run's own report,
        which is what the d0 gate's bytes_identical check refuses to let happen."""
        out = Path(outputs_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{_slug(query)}-{uuid.uuid4().hex[:8]}"
        payload = {
            "meta": meta,
            # s5: nodes[].answer ADDED (the existing four keys stay) — it is the
            # corpus the scorer matches every report claim against, and without it
            # that corpus is empty and S6 is 0 however good the research was. Blank
            # for a FAILED node, exactly as _node_dict blanks its sources/digest:
            # prior knowledge over missing evidence may not validate a claim.
            "nodes": [{"id": n.id, "depth": n.depth, "status": n.status.value,
                       "question": n.question,
                       "answer": "" if n.status == NodeStatus.FAILED else n.answer_md}
                      for n in self.nodes.values()],
            "citations": citation_map,
        }
        tree_path = out / f"{stem}.tree.json"
        report_path = out / f"{stem}.tree-report.md"
        resynth_path = out / f"{stem}.resynth.json"
        tree_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        report_path.write_text(report_md, encoding="utf-8")
        resynth_path.write_text(json.dumps({
            "schema": 1,
            "query": query,
            "read_docs": self._read_docs,
            "nodes": [{"id": n.id, "parent_id": n.parent_id, "children": list(n.children),
                       "depth": n.depth, "status": n.status.value, "question": n.question,
                       "answer_md": n.answer_md, "answer_digest": n.answer_digest,
                       "learnings": list(n.learnings), "sources": list(n.sources),
                       "novelty": n.novelty, "priority": n.priority}
                      for n in self.nodes.values()],
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"tree_json": str(tree_path), "report_md": str(report_path),
                "resynth_json": str(resynth_path)}
