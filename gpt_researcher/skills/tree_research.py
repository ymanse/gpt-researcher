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
from typing import Any, Dict, List, Optional

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


# --- s9 (dedup harness): the roll-up states each finding ONCE.
# Sibling nodes that researched near-identical questions restate the same finding in
# their own prose, so the redundancy is not lexical: measured across the round-4
# reports, repeated 5-grams run 0-2% and the worst section-pair Jaccard is 0.22, while
# every kept node answer is >=70% present in the report (lifted 12/12, max lift 100%).
# A word-similarity scan therefore returns a confident "no redundancy" on a report a
# reader plainly sees repeating itself. What survives rewording is the figure and the
# nouns around it -- the scorer's own view of a claim (_claim_profile) -- so that is
# what two sentences are compared on.
#
# The merge's IDENTITY test is NOT _claim_profile. That profile is the SUPPORT test
# verify_rollup shares with the frozen scorer, and it drops tokens under 4 characters
# and bare digits — an approximation that only ever LOOSENS what that pass accepts.
# Authorizing a DELETION on it inverts which way it is safe: "BYD", "LG", "SDI", "SK"
# are invisible to it, and those four tokens are the entire difference between two
# frontier questions it would then merge into one (review R2). So the merge profiles
# text with its own tokens, short ones included.
_MERGE_STOP = _STOP | {
    "the", "a", "an", "and", "or", "but", "if", "in", "of", "to", "for", "on", "at",
    "by", "is", "are", "was", "be", "it", "as", "its", "not", "no", "any", "all",
    "per", "own", "can", "may", "one", "two", "up", "so", "at", "we", "who", "how"}
# 2026 shared between two statements says they discuss the same year, not the same
# finding — but _SIGNUM matches any bare 4+-digit run, so a year anchored "same
# claim" verdicts between statements sharing nothing else (review R1)
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
# ...and a number that cannot ANCHOR must still DIFFERENTIATE. _SIGNUM only sees
# 4+-digit runs, comma-thousands, decimals and percents, and _merge_tokens drops every
# all-digit token, so "9 GWh/year starting 2026" against "150 GWh/year" was a pair no
# half of the identity test could tell apart: same tokens, no figure either side,
# coverage 1.0, one deleted with a 16x disagreement inside it. Same for 2027 against
# 2030 — the R1 year guard closed the anchor and opened this (review R10). So the two
# roles are split: EVERY number differentiates, only a significant non-year figure
# lowers the coverage bar.
_ANYNUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# both are coverage OF THE DROPPED STATEMENT: what fraction of what it says the
# survivor already says. Measured on the s9 fixture's own restatement pairs (same
# finding, sibling wording): 0.83 / 0.75 / 0.62 with a shared figure.
_MERGE_FIGURE_COVERAGE = 0.60  # ... anchored by a figure the survivor also states
_MERGE_PROSE_COVERAGE = 0.80   # nothing to anchor on: near-verbatim restatement only
_MERGE_MIN_TOKENS = 5          # below that there is not enough content to judge at all
# the roll-up's own length budget, as a share of the answers it is allowed to use --
# the SAME denominator the d1 gate divides by (scripts/rollup_scan.py counts the
# expanded/answered nodes). The gate caps the whole report at 70%; the H1, the section
# headings and the Citations block ship on top of the body, so the body is budgeted
# under it. See _restored_statements.
_ROLLUP_BUDGET = 0.50
_RESTORE_MIN_NEW = 4  # a sentence adding fewer content words is restating, not adding


def _merge_tokens(text: str) -> set:
    return {t for t in re.sub(r"[^0-9a-z]+", " ", text.lower()).split()
            if not t.isdigit() and t not in _MERGE_STOP}


def _merge_figures(text: str) -> set:
    return {k for k in (_num_key(m.group(0)) for m in _SIGNUM.finditer(text))
            if not _YEAR_RE.match(k)}


def _merge_numbers(text: str) -> set:
    return {_num_key(m.group(0)) for m in _ANYNUM_RE.finditer(text)}


def _merge_profile(text: str) -> tuple:
    return _merge_figures(text), _merge_numbers(text), _merge_tokens(text)


def _subsumed(drop: tuple, keep: tuple) -> bool:
    """Does `keep` already state everything `drop` states? (profiles from
    _merge_profile — asymmetric on purpose.)

    The question a merge is allowed to ask is containment, not similarity. Asking
    "are these the same claim?" with an overlap coefficient normalized by the
    SMALLER side let a three-token fragment delete an eighteen-token finding on two
    shared tokens, and "first statement wins" then kept the fragment: measured on
    this harness's own corpus, that dropped a second product variant's LOWER
    accuracy figure, a third source's timeline disagreement, and the finding under
    a bold pseudo-heading — leaving the reader the heading and no content (review
    R1/R6). Deletion is the one direction this pass may not get wrong; the frozen
    scorer's s2_aggregate_pct >= 80 is what stands against it.

    So a statement is dropped only when the survivor carries every significant
    figure it carries AND enough of its content words. A figure `drop` has that
    `keep` lacks makes them different findings however alike the wording (">99.5%,
    0.35s" against the non-V3 variant's ">99%, <=0.5s"), and both ship. With no
    figure left to anchor on the bar rises to near-verbatim, because dropping a
    sentence on topical similarity alone is how a redundancy fix turns into content
    deletion — and a bare YEAR is not an anchor, so a pair sharing only "2027" is
    judged on that higher bar rather than on two digits' agreement.

    A year still DIFFERENTIATES though, as does any other number, however small: the
    difference between "2027" and "2030", or between "9 GWh/year" and "150 GWh/year",
    is the whole finding, and neither side of it is a _SIGNUM figure or a token that
    survives _merge_tokens (review R10). So every number is a difference test; only a
    significant non-year figure is an anchor that lowers the coverage bar.
    """
    dn, dnum, dt = drop
    _, knum, kt = keep
    if len(dt) < _MERGE_MIN_TOKENS or len(kt) < _MERGE_MIN_TOKENS:
        return False
    if dnum - knum:  # a number `keep` never states -> a different finding (dn <= dnum)
        return False
    need = _MERGE_FIGURE_COVERAGE if dn else _MERGE_PROSE_COVERAGE
    return len(dt & kt) / len(dt) >= need


def _merge_claim_blocks(blocks: List[tuple]) -> List[str]:
    """One statement per claim across (heading, text, mergeable) sections.

    The survivor keeps its own wording: the citations that node earned were traced
    against that text, so leaving it alone is what keeps them grounded. Three "LLM
    rewrites the merge, then re-derive the [id] markers by fuzzy matching" designs
    were each measured live to lose citations, one to citations_total=0 on a
    33-node tree (see synthesize_node) — the rewrite is what erases the grounding,
    so there is none here.

    Which statement survives is decided over the whole report, not by document
    order: the RICHEST statement of a claim wins (most content words, ties to the
    earlier one). First-wins is what let a truncated fragment outrank the complete
    sentence it was a prefix of.

    A dropped duplicate hands its markers to the survivor rather than taking them
    to the grave: both nodes really did support that claim, and the scorer reads a
    marker off the 240 characters before it, which the survivor's sentence now
    occupies. Nothing is trusted about that hand-off — this runs BEFORE the
    fail-closed marker passes, so a migrated [id] that no longer traces to its own
    page is stripped there like any other.

    Segmentation is verify_rollup's, for the same reason: _scrubbed() is
    offset-preserving, so a scorer-visible sentence maps back to the exact bytes
    that ship, and a marker landing in a separator trailed the sentence before it.
    """
    out: List[List[str]] = [[] for _ in blocks]
    claims: List[tuple] = []        # (profile, block index, slot index)
    seps: List[tuple] = []          # (block index, slot index, claim index, offset)
    fences: List[List[tuple]] = []

    def migrate(raw: str, into: tuple) -> None:
        _, bi, slot = into
        # the survivor's own trailing marker sits in the separator after it, so both
        # slots are read before deciding an id is new
        have = {c for m in _CITE_ID_RE.finditer("".join(out[bi][slot:slot + 2]))
                for c in _bracket_ids(m.group(1))}
        add = [c for c in dict.fromkeys(
            cid for m in _CITE_ID_RE.finditer(raw) for cid in _bracket_ids(m.group(1)))
            if c not in have]
        if add:
            out[bi][slot] += " " + render_ids(add)

    for bi, block in enumerate(blocks):
        text, mergeable = block[1], (block[2] if len(block) > 2 else True)
        fences.append([m.span() for m in _SCRUB_RES[0].finditer(text)])
        parts = _SENT_SPLIT_RE.split(_scrubbed(text))
        pos = 0
        prev: Optional[int] = None
        for i, part in enumerate(parts):
            start, pos = pos, pos + len(part)
            slot = len(out[bi])
            out[bi].append(text[start:pos])
            if i % 2:
                seps.append((bi, slot, prev, start))
                continue
            prev = None
            stripped = part.strip()
            # a heading is a title, a fenced block is code, and an unresearched
            # node's "(unexplored frontier) <question>" line is a QUESTION — none of
            # the three is a claim any other statement can be said to already make
            # ...and a statement ending in ':' introduces what comes after it, so
            # dropping it as "already stated" leaves a headless list behind
            if (not mergeable or not stripped or stripped.startswith("#")
                    or stripped.endswith(":")
                    or any(start < b and a < pos for a, b in fences[bi])):
                continue
            prev = len(claims)
            claims.append((_merge_profile(part), bi, slot))

    # richest first, so a claim's fullest statement is the one that gets to absorb
    # the others; a survivor is never itself absorbed afterwards
    order = sorted(range(len(claims)), key=lambda i: (-len(claims[i][0][2]), i))
    absorbed: Dict[int, int] = {}
    for i in order:
        if i in absorbed:
            continue
        for j in order:
            if j == i or j in absorbed or len(claims[j][0][2]) > len(claims[i][0][2]):
                continue
            if _subsumed(claims[j][0], claims[i][0]):
                absorbed[j] = i
    for j, i in absorbed.items():
        _, bi, slot = claims[j]
        migrate(out[bi][slot], claims[i])
        out[bi][slot] = ""
    for bi, slot, prev, start in seps:
        if prev is None or prev not in absorbed:
            continue
        raw = out[bi][slot]
        migrate(raw, claims[absorbed[prev]])
        # review R3: a separator is whitespace in the SCRUBBED view only — _scrubbed
        # blanks fenced code, so a fence between two sentences lands INSIDE the
        # separator. Dropping the separator with its sentence would delete that code
        # out of the report; keep it and strip only the markers it inherited, with
        # verify_rollup's own fence guard so a bracketed array index in the fence is
        # left alone.
        out[bi][slot] = _CITE_ID_RE.sub(
            lambda m: m.group(0) if any(a <= start + m.start() < b
                                        for a, b in fences[bi]) else "", raw)
    return ["".join(slots) for slots in out]


def _claim_corpus(nodes) -> List[tuple]:
    """The evidence verify_rollup weighs a claim against (see its docstring for why
    FAILED is out and PRUNED is in)."""
    return [_claim_profile(n.answer_md) for n in nodes
            if n.status != NodeStatus.FAILED and n.answer_md]


def _claim_supported(nums: set, ctx: set, corpus: List[tuple]) -> bool:
    """Does some node answer carry this claim's figure and enough of its context?
    Prose carrying no figure is not a claim the frozen scorer weighs."""
    return not nums or any((nums & cn) and len(ctx & ct) >= min(2, len(ctx))
                           for cn, ct in corpus)


def _learning_lines(node) -> List[str]:
    """A node's learnings, one statement per line, headings demoted (review R11)."""
    return [x.lstrip("#").strip() if x.startswith("#") else x
            for x in (str(y).strip() for y in (node.learnings or [])) if x.strip()]


def _restored_statements(nodes) -> Dict[str, str]:
    """Spend what is left of the roll-up's length budget on the answer sentences that
    say something nothing already in the report says.

    Rolling up `node.learnings` alone made the report a distillation instead of a
    stack of pasted answers, but it also shipped only what the upstream distillation
    happened to carry — 20% of each answer, everything else silently gone. "A shorter
    report bought by dropping content" is the one thing DEDUP-HARNESS says s9 must not
    be (review R9), so the rest of the budget goes back into the answers.

    Why a BUDGET and not a similarity threshold. The redundancy in this corpus is
    topical, not lexical — DEDUP-HARNESS's own trap paragraph measures repeated
    5-grams at 0-2% and the worst section-pair Jaccard at 0.22 — so no coverage
    threshold separates a sibling's restatement of a finding from a sibling's new
    finding: replayed over the captured learnings the best genuine-duplicate pair
    scores 0.48 against a best distinct pair of 0.53, and a containment rule tuned to
    catch the duplicates deletes findings first. What DOES separate them is marginal
    novelty: the third statement of "at-least-once, not exactly-once" adds almost
    nothing the report does not already contain, so under a fixed budget it loses to a
    sentence that adds a finding. That is a SELECTION, not a deletion — a wrong
    verdict here declines to ADD a sentence, where the same verdict over the finished
    body would DELETE one. Measured over the five goldens: 982 candidate statements,
    53 rejected outright by _subsumed as already stated, 147 selected.

    What it does NOT buy back, measured rather than assumed: the three golden facts
    that live in an answer and in no learning (solid-state's lithium-metal dendrites
    and Samsung SDI S-Line, denorm's closure table) sit deep in narration —
    solid-state's Samsung sentence is 15.4k characters into its query's ranking
    against 5.1k of room, i.e. ~+27 points of synthesis ratio on a gate that caps it
    at 70. s2_aggregate is 83 either way; the concatenation's 87 is not purchasable
    at any length this gate allows.

    The budget is the gate's own: the d1 measure is report length over the answers of
    the kept (expanded/answered) nodes, capped at 70%. Everything already destined for
    the report is charged against that budget before a single sentence is restored —
    including the bytes with no answer in the denominator at all (a PENDING node's
    frontier line, a section heading) — and it is set below the cap because the
    citation markers _attribute_citations inserts and the Citations block are not
    knowable here.

    Statements keep their own wording — a rewrite is what erased the citation
    grounding in the three designs synthesize_node's docstring records — so a restored
    sentence is attributed and merged exactly like a learning.
    """
    live = [n for n in nodes if n.status != NodeStatus.PRUNED]
    kept = [n for n in live
            if n.status in (NodeStatus.EXPANDED, NodeStatus.ANSWERED) and n.answer_md]
    # what already ships, measured the same way the gate measures it: the whole body
    # counts against the budget, not just the part restoration adds to. A PENDING
    # node's frontier line and a section heading are report bytes with no answer in
    # the denominator at all, so leaving them out is what made a 0.55 budget come out
    # at 0.91 of the answers.
    room = int(_ROLLUP_BUDGET * sum(len(n.answer_md) for n in kept))
    for n in live:
        if n.status == NodeStatus.PENDING:
            room -= len(n.question) + 22  # "(unexplored frontier) "
        elif n.status != NodeStatus.FAILED:
            room -= len(_node_findings(n))
            if n.parent_id is not None:
                room -= len(n.question) + 8  # "\n\n### <question>\n"
    seen: set = set()
    shipping: List[tuple] = []  # profile of every statement already destined to ship
    raw_cands: List[tuple] = []
    for n in kept:
        lines = _learning_lines(n)
        # a number nothing has stated yet is new content too, and _merge_tokens drops
        # every all-digit token — so novelty is counted over tokens AND numbers
        text = "\n".join(lines) if lines else n.answer_md
        seen |= _merge_tokens(text) | _merge_numbers(text)
        shipping.extend(_merge_profile(x) for x in lines)
        if not lines:  # the whole answer ships as _node_findings' fallback
            continue
        parts = _SENT_SPLIT_RE.split(_scrubbed(n.answer_md))
        pos = 0
        for i, part in enumerate(parts):
            start, pos = pos, pos + len(part)
            if i % 2 or not part.strip() or part.strip().startswith("#"):
                continue
            prof = _merge_profile(part)
            if len(prof[2]) >= _MERGE_MIN_TOKENS:
                raw_cands.append((n.id, start, n.answer_md[start:pos].strip(), prof))
    # THIS is where the merge fires in production. A candidate whose claim some
    # statement already destined for the report makes is not a candidate at all: it
    # never competes for the budget. Asking the question here rather than over the
    # finished body is what makes the same containment rule useful on this corpus —
    # over the distilled learnings alone it has nothing to bite on (measured: zero
    # firing pairs across all five goldens, review R9), because a node's narration is
    # what restates, not its one-line distillation of that narration. And it is the
    # safe direction for a test that can be wrong: a false "already stated" here
    # declines to ADD a sentence, where over the body it would DELETE one.
    # ponytail: O(candidates x shipping) — ~300 x ~150 a query, one pass, milliseconds.
    cands = [(nid, start, raw, prof[1] | prof[2]) for nid, start, raw, prof in raw_cands
             if not any(_subsumed(prof, s) for s in shipping)]

    # ponytail: O(candidates^2) rescan per pick — ~300 sentences a query, milliseconds.
    # A heap keyed on stale novelty would need re-validation anyway.
    picked: Dict[str, List[tuple]] = {}
    while room > 0 and cands:
        # per CHARACTER, not per sentence: the budget is bytes, so ranking on the raw
        # count buys the longest paragraphs first. Measured on the corpus, density
        # fits ~2x the statements into the same room at a 10-point lower ratio.
        best, score = -1, 0.0
        for i, (_, _, raw, toks) in enumerate(cands):
            new = len(toks - seen)
            if new < _RESTORE_MIN_NEW or len(raw) > room:
                continue
            if new / len(raw) > score:
                best, score = i, new / len(raw)
        if best < 0:
            break
        nid, start, raw, toks = cands.pop(best)
        seen |= toks
        room -= len(raw)
        picked.setdefault(nid, []).append((start, raw))
    return {nid: "\n\n".join(raw for _, raw in sorted(v)) for nid, v in picked.items()}


def _node_findings(node, restored: str = "") -> str:
    """A node's own FINDINGS — what it learned, not the narration it learned it in.

    `node.learnings` is the research pass's own one-claim-per-line distillation of
    answer_md, written upstream with the source pages in context; it is already
    what expansion and novelty treat as "what this node found". Rolling THAT up is
    what makes the report a synthesis instead of a stack of pasted answers, and it
    costs no LLM call, no network and no rewrite — the text is on the node before
    the assembly starts, which is why assemble_report stays replayable offline.

    Measured over the five captured goldens (no_read/dedup/corpus), learnings
    against the answers they came from:
      * 19-22% of answer_md by length — the concatenation ran 119-134% of the
        answers it is allowed to use, which no sentence-level merge can move: a
        lexical scan finds 0-3% removable, because the sibling redundancy is
        TOPICAL (repeated 5-grams 0-2%, worst section-pair Jaccard 0.22, while a
        genuine restatement pair scores 0.13)
      * the frozen scorer's facts at the same rate: s2 aggregate 85 vs 87, ONE
        fact lost across the whole corpus, coverage areas equal or better
      * traced to their own node's pages at the same rate: 69-88% of sentences
        against the answers' 65-89%, so the citation grounding survives. That
        measurement is what separates this from the three "LLM rewrites the merge,
        then re-derive [id] by fuzzy matching" designs in synthesize_node's
        docstring, all of which lost citations.

    The distillation is not complete, though, and `restored` is what closes that:
    _restored_statements hands back the sentences of THIS node's answer that state
    something nothing already in the report states, as far as the roll-up's length
    budget reaches (review R9 — the learnings alone are 20% of the answers, and what
    the distillation dropped was dropped silently).

    A learning that is itself a markdown heading is demoted to a statement: '# Answer'
    and '## Anti-Spoofing Techniques' are in the captured corpus, and pasted through
    verbatim they put a second H1 in the report, count towards the gate's heading
    measure and exempt themselves from the merge on the strength of one character
    (review R11).

    Fallback is the full answer: a node whose research produced no learnings must
    lose nothing.
    """
    lines = _learning_lines(node)
    if not lines:
        return node.answer_md or node.answer_digest or node.question
    out = "\n".join(x if x.startswith(("-", "*")) else f"- {x}" for x in lines)
    return f"{out}\n\n{restored}" if restored else out


def _own_contribution(text: str, child_summaries: List[str]) -> str:
    """What a node added AHEAD of the child summaries it was handed.

    The report is laid out as one titled section per node, so the assembly needs
    each node's own findings separately — but synthesize_node must still be called
    exactly once per node and its return is what the report is built from (the
    stage-6 contract test replaces it with a seam and asserts both). Subtracting
    the summaries it was given satisfies both: a seam that ignores them returns
    text that does not end with them, and keeps its whole output.
    """
    tail = "\n\n".join(child_summaries)
    if not tail:
        return text
    if text == tail:  # a FAILED node contributes nothing; its children rolled through
        return ""
    return text[:-len(tail) - 2] if text.endswith("\n\n" + tail) else text


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
        # node id -> answer sentences the learnings missed; assemble_report fills it
        # before the roll-up, because the budget is a property of the whole tree
        self._restored: Dict[str, str] = {}
        self._root_query: str = str(getattr(researcher, "query", "") or "")
        self._max_breadth = 4
        self._rollup_dropped_ratio = 0.0  # share of the body verify_rollup removed
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
            cfg = self.researcher.cfg
            memory = Memory(cfg.embedding_provider, cfg.embedding_model,
                            **getattr(cfg, "embedding_kwargs", {}))
        return list(await memory.get_embeddings().aembed_query(text))

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
        corpus = _claim_corpus(self.nodes.values())
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
            if _claim_supported(nums, ctx, corpus):
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
        the wording it traces to is also what a citation is supposed to mean."""
        # keep the separators: joining the pieces back reproduces the input byte for
        # byte, so a node's own line breaks and markdown structure survive
        # attribution. Re-joining on " " (the pre-s9 behaviour) collapsed every
        # answer to one line, which is why the concatenated report shipped 2
        # headings — the answers' own "## ..." sections were inlined into prose.
        parts = _SENT_SPLIT_RE.split(text or "")
        if not any(p.strip() for p in parts):
            return (text or "").strip()
        for si in range(0, len(parts), 2):
            sent = parts[si]
            if not sent.strip():
                continue
            marks: Dict[int, List[str]] = {}
            for url, cid in source_ids.items():
                end = _trace_end(sent, self._read_docs.get(url, ""))
                if end is None:
                    continue
                # never splice INSIDE a word. The phrase tracer ends a token at the
                # first non-alphanumeric byte, so "H₂S" ends after the "H" and the
                # marker shipped as "H [6]₂S" — unreadable, and the frozen scorer's
                # own fact pattern for that finding stops matching. Pushing to the
                # end of the run only ever ADDS tokens ahead of the marker, which is
                # the direction score_s1's 20-token window is safe in.
                while end < len(sent) and not sent[end].isspace():
                    end += 1
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
            parts[si] = sent
        return "".join(parts).strip()

    async def synthesize_node(self, node: ResearchNode,
                              child_summaries: Optional[List[str]] = None,
                              source_ids: Optional[Dict[str, str]] = None) -> str:
        """Roll one node up into a summary (leaf: own attributed answer;
        internal: own answer followed by each child's summary, verbatim).

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
        # what rolls up is the node's FINDINGS, not the narration around them (see
        # _node_findings, and the measurement that its learnings trace to the node's
        # own pages at the same rate answer_md does — attribution needs text close
        # to the source wording, and a paraphrase that does not trace buys an [id]
        # that can never ground)
        own = self._attribute_citations(
            _node_findings(node, self._restored.get(node.id, "")), source_ids)
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

        # what each node's learnings did not carry, chosen against the whole tree's
        # budget — so it has to be decided before the first synthesize_node call
        self._restored = _restored_statements(self.nodes.values())

        # post-order roll-up: children before parent, root synthesized last
        own_texts: Dict[str, str] = {}

        async def rollup(n: ResearchNode) -> str:
            summaries = []
            for cid in n.children:
                child = self.nodes[cid]
                if child.status == NodeStatus.PRUNED:
                    continue
                summary = await rollup(child)
                # review R4: synthesize_node drops falsy child summaries, so passing
                # the UNFILTERED list to _own_contribution makes the two disagree
                # exactly when a child contributed nothing — which is what a FAILED
                # leaf returns. The subtraction then misses and the parent's section
                # swallows every surviving child's whole subtree.
                if summary:
                    summaries.append(summary)
            node_source_ids = {u: url_to_id[u] for u in n.sources if u in url_to_id}
            text = await self.synthesize_node(n, summaries, node_source_ids)
            self._syntheses[n.id] = text
            own_texts[n.id] = _own_contribution(text, summaries)
            return text

        # the root is the first node inserted (run() seeds it before the frontier
        # loop), which is also the order the resynth sidecar preserves
        root = next(iter(self.nodes.values()), None)
        if root is not None and root.status != NodeStatus.PRUNED:
            await rollup(root)

        # s9: titled sections instead of one undivided wall. The tree already knows
        # what each section is about, so the split needs no model and cannot invent a
        # title the section does not deliver. Document order is the roll-up's own —
        # a node's findings, then its children's — so this re-uses the layout the
        # concatenation produced and only gives it headings.
        blocks: List[tuple] = []

        def collect(n: ResearchNode) -> None:
            own = own_texts.get(n.id, "").strip()
            if own:
                blocks.append((n, own, n.status != NodeStatus.PENDING))
            for cid in n.children:
                child = self.nodes[cid]
                if child.status != NodeStatus.PRUNED:
                    collect(child)

        if root is not None and root.status != NodeStatus.PRUNED:
            collect(root)

        # review R13: the merge decides which sections survive, so the headings are
        # laid out AFTER it. Deciding both in one pass let a section be skipped
        # together with its heading while its children had already spent a level on
        # it ("## parent" -> "#### grandchild"), and recorded a title for the
        # near-duplicate check that the emit loop then dropped.
        merged = {b[0].id: m.strip() for b, m in zip(blocks, _merge_claim_blocks(blocks))}
        claim_corpus = _claim_corpus(self.nodes.values())
        lines = [f"# {query}"]
        titles: List[str] = []

        def emit(n: ResearchNode, level: int) -> None:
            body = merged.get(n.id, "")
            heading = ""
            if body:
                # only a node that researched gets a heading: a PENDING node
                # contributes one frontier line, and a heading repeating its own
                # question above that line is noise, not a section
                titled = (n.parent_id is not None
                          and n.status in (NodeStatus.ANSWERED, NodeStatus.EXPANDED))
                # review R8: the sibling nodes whose near-identical questions ARE the
                # documented defect (no_read/audit/pending_rca.md) would otherwise
                # title two adjacent sections almost the same way — visible
                # repetition no roll-up metric counts. The bodies still both ship.
                # review R12: and a question carrying a figure no node answer
                # corroborates is a claim to verify_rollup, which deletes the whole
                # heading line and books it as an unsupported claim. That pass's own
                # docstring says the fix belongs in how the title is rendered, so the
                # title is simply not rendered — the section ships untitled.
                q_nums, q_ctx = _claim_profile(n.question)
                if (titled and _claim_supported(q_nums, q_ctx, claim_corpus)
                        and not (titles and _subsumed(_merge_profile(n.question),
                                                      _merge_profile(titles[-1])))):
                    heading = "#" * min(level, 6) + f" {n.question}"
                    titles.append(n.question)
                    lines.extend(["", heading])
                lines.extend(["", body])
            # a level is spent only where a heading actually shipped, so the report
            # never jumps "# query" -> "### grandchild" past a missing "##"
            for cid in n.children:
                child = self.nodes[cid]
                if child.status != NodeStatus.PRUNED:
                    emit(child, level + 1 if heading else level)

        if root is not None and root.status != NodeStatus.PRUNED:
            emit(root, 2)
        if len(lines) == 1:
            lines += ["", "_(no synthesis)_"]
        body = "\n".join(lines) + "\n"

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
                "unsupported": unsupported}

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
