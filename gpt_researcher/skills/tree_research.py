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
_ITEM = r"(?:\d+|\d{1,3}\s*-\s*\d{1,3})"
_SEP = r"(?:\s*[,;]+\s*|\s+)"
_CITE_ID_RE = re.compile(rf"\[\s*({_ITEM}(?:{_SEP}{_ITEM})*){_SEP}?\s*\]")
# range alternative FIRST so a spaced range ("1 - 3") stays one token — splitting
# the group on _SEP instead shreds a range at its own internal whitespace
_TOKEN_RE = re.compile(r"(\d{1,3})\s*-\s*(\d{1,3})|(\d+)")


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


def find_uncited_ids(report_md: str, citation_map: Dict[str, str]) -> List[str]:
    """[id] markers in report_md with no citations-map entry, first-appearance order."""
    out: List[str] = []
    for m in _CITE_ID_RE.finditer(report_md or ""):
        for cid in _bracket_ids(m.group(1)):
            if cid not in citation_map and cid not in out:
                out.append(cid)
    return out


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
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# the scorer blanks these before it splits sentences, so a marker or URL never
# contributes a "number" or a context token
_SCRUB_RES = (re.compile(r"\[\d{1,3}\]"), re.compile(r"\(https?://\S+\)"),
              re.compile(r"https?://\S+"))
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
    """(significant numbers, context tokens) — the scorer's view of one text."""
    for rx in _SCRUB_RES:
        text = rx.sub(" ", text)
    return {_num_key(m.group(0)) for m in _SIGNUM.finditer(text)}, _ctx_tokens(text)


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
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher answering one focused question from collected context."},
                {"role": "user",
                 "content": (
                     f"Question: {node.question}\n\nContext:\n{context}\n\n"
                     "Write three sections:\n"
                     "ANSWER: a markdown answer (<=400 words). Do not add citation "
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
                     + f"\n\nGenerate up to {self._max_breadth} follow-up questions that carry "
                       "the root research query into ground the list above does NOT yet cover. "
                       "Target what is missing, not variations of what was already found. Each "
                       "must be disjoint from the others and from the covered questions. If you "
                       "know the specific project, product, company, standard, or author that "
                       "originated this topic, name that entity by its proper name in the "
                       "question itself (e.g. 'What does the <project>'s own blog/documentation "
                       "say about X' rather than a generic phrasing of the same question), so "
                       "the search targets that primary source directly instead of generic "
                       "secondary commentary. Return "
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
        such boundary to get wrong, and mirrors the shape of the S1 scorer's own
        near-marker check this whole pass exists to satisfy."""
        def _check(m: "re.Match[str]") -> str:
            window = _CITE_ID_RE.sub(" ", body[max(0, m.start() - 300):m.start()])
            kept = [cid for cid in _bracket_ids(m.group(1))
                    if text_supported(window, self._read_docs.get(citation_map.get(cid, ""), ""))]
            return f"[{', '.join(kept)}]" if kept else ""
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
        """
        corpus = [_claim_profile(n.answer_md) for n in self.nodes.values()
                  if n.status != NodeStatus.FAILED and n.answer_md]
        body = body or ""
        fenced = [m.span() for m in _FENCE_RE.finditer(body)]
        parts = _SENT_SPLIT_RE.split(body)
        contradictions: List[str] = []
        unsupported: List[str] = []
        out: List[str] = []
        pos = 0
        for i, part in enumerate(parts):
            start, pos = pos, pos + len(part)
            # odd indices are the separators; the scorer ignores fenced code
            if i % 2 or not part.strip() or any(s < pos and start < e for s, e in fenced):
                out.append(part)
                continue
            nums, ctx = _claim_profile(part)
            if not nums:  # prose carrying no figure is not a claim the scorer weighs
                out.append(part)
                continue
            need = min(2, len(ctx))
            if any((nums & cn) and len(ctx & ct) >= need for cn, ct in corpus):
                out.append(part)
                continue
            if any(cn and len(ctx & ct) >= 3 and not (nums & cn) for cn, ct in corpus):
                contradictions.append(part.strip())
            else:
                unsupported.append(part.strip())
        return "".join(out), contradictions, unsupported

    def _attribute_citations(self, text: str, source_ids: Dict[str, str]) -> str:
        """Attach each node source's global [id] only to the sentences it actually
        supports, instead of dumping every node source in one trailing block after
        the whole answer. The S1 grounding scorer requires an [id] marker to sit
        next to text that literally traces to that source; a bulk trailing dump
        (the pre-fix behavior) attaches every source to whichever sentence happens
        to be last, so most markers end up ungrounded even when the underlying
        source really does support SOME sentence in the text."""
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return (text or "").strip()
        out = []
        for sent in sentences:
            ids = sorted((cid for url, cid in source_ids.items()
                         if text_supported(sent, self._read_docs.get(url, ""))),
                        key=int)
            cites = " ".join(f"[{c}]" for c in ids)
            out.append(f"{sent} {cites}".strip() if cites else sent)
        return " ".join(out)

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

    # -------------------------------------------------------------------- run

    async def run(self, query: Optional[str] = None, max_depth: int = 3,
                  max_breadth: int = 4, max_nodes: int = 40,
                  token_budget: int = 300_000, credit_budget: float = 150.0,
                  novelty_threshold: float = 0.30, expansion_policy: str = "best_first",
                  stream: bool = False, outputs_dir: Optional[str] = None,
                  time_budget_s: float = 600.0) -> Dict[str, Any]:
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

        # budgets are checked BEFORE each frontier pop; accepted-but-unresearched
        # nodes stay PENDING in the tree ("unexplored frontier")
        # ponytail: nodes are researched strictly sequentially (~45s each), so
        # max_nodes alone can't bound wall-clock — the caller's MCP idle timeout
        # fires first. time_budget_s caps expansion only; the sequential roll-up
        # that follows adds ~0.8x on top (measured: 180s expansion -> 321s total),
        # so total ~= 1.8 * time_budget_s. The 600s default lands near 1080s,
        # inside a 1200s idle timeout. Parallelize the frontier before raising it.
        while (len(frontier) and researched < max_nodes
               and self.tokens_spent < token_budget
               and self.credits_spent < credit_budget
               and time.time() - start < time_budget_s):
            node = frontier.pop()
            try:
                await self.research_node(node)
            except Exception as e:
                logger.error(f"tree node {node.id} research failed: {e}")
                node.status = NodeStatus.FAILED
                continue
            researched += 1
            self.tokens_spent += node.tokens_spent
            self.credits_spent += node.credits_spent

            if node.status == NodeStatus.FAILED:
                # defect 3: nothing to expand from and nothing to roll up. Skipping
                # compute_novelty matters too — registering a starved node's
                # question as covered ground would let the hole it left prune a
                # later child that would have filled it.
                continue

            node.novelty = self.compute_novelty(node)
            if node.novelty < novelty_threshold:
                node.status = NodeStatus.PRUNED
                pruned += 1
                continue  # pruned nodes are never expanded

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
                    child.priority = max(0.0, 0.5 + 0.15 * node.priority - 0.10 * child.depth)
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

        # post-order roll-up: children before parent, root synthesized last
        async def rollup(n: ResearchNode) -> str:
            summaries = []
            for cid in n.children:
                child = self.nodes[cid]
                if child.status == NodeStatus.PRUNED:
                    continue
                summaries.append(await rollup(child))
            node_source_ids = {u: url_to_id[u] for u in n.sources if u in url_to_id}
            text = await self.synthesize_node(n, summaries, node_source_ids)
            self._syntheses[n.id] = text
            return text

        root_text = "" if root.status == NodeStatus.PRUNED else await rollup(root)

        body = "\n".join([f"# {query}", "", root_text or "_(no synthesis)_", ""])

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
                return f"[{', '.join(kept)}]" if kept else ""

            body = _CITE_ID_RE.sub(_keep_cited, body)

        body = self._prune_ungrounded_markers(body, citation_map)

        # defect 6b: last, so the claim check sees the body as it will ship and the
        # Citations list below is rendered from what SURVIVED it — a dropped claim
        # must not leave its source behind in the citations block.
        body, contradictions, unsupported = self.verify_rollup(body)
        if contradictions or unsupported:
            logger.error(f"roll-up consistency: dropped {len(contradictions)} claim(s) "
                         f"contradicting a node answer and {len(unsupported)} no node "
                         f"answer supports: {[*contradictions, *unsupported]}")

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
        """Write tree.json (smoke-gate contract shape) + final report markdown."""
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
        tree_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        report_path.write_text(report_md, encoding="utf-8")
        return {"tree_json": str(tree_path), "report_md": str(report_path)}
