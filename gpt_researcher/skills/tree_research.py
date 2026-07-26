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


_CITE_ID_RE = re.compile(r"\[(\d+)\]")


def find_uncited_ids(report_md: str, citation_map: Dict[str, str]) -> List[str]:
    """[id] markers in report_md with no citations-map entry, first-appearance order."""
    out: List[str] = []
    for m in _CITE_ID_RE.finditer(report_md or ""):
        cid = m.group(1)
        if cid not in citation_map and cid not in out:
            out.append(cid)
    return out


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
        self._read_docs: Dict[str, str] = {}  # url -> text of documents actually scraped
        self._seen_learnings: set = set()
        self._syntheses: Dict[str, str] = {}
        self._max_breadth = 4
        self.tokens_spent = 0
        self.credits_spent = 0.0

    # ------------------------------------------------------------------ seams
    # Each of these is a deterministic-test seam: unit tests replace them on the
    # instance, so run() must route every LLM/embedding/research call through them.

    async def research_node(self, node: ResearchNode) -> None:
        """Research one node with a dedicated GPTResearcher; mutates the node."""
        node.status = NodeStatus.RESEARCHING
        before = set(self.visited_urls)
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
        fresh = set(researcher.visited_urls) - before
        node.sources.extend(sorted(fresh if fresh else set(researcher.visited_urls)))
        try:
            self.visited_urls.update(researcher.visited_urls)
        except (AttributeError, TypeError):
            pass

        if isinstance(context, list):
            context = "\n\n".join(str(c) for c in context)
        context = str(context or "")[:60000]

        # ponytail: one LLM call yields answer + digest + learnings; parse-tolerant
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher answering one focused question from collected context."},
                {"role": "user",
                 "content": (
                     f"Question: {node.question}\n\nContext:\n{context}\n\n"
                     "Write three sections:\n"
                     "ANSWER: a cited markdown answer (<=400 words).\n"
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
            m = re.match(r"^\s*(ANSWER|DIGEST|LEARNINGS)\s*:\s*(.*)$", line)
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

        # defect 6a: node.sources means "read AND quoted", not "every retriever
        # return". Keep only URLs whose scraped document supports some sentence
        # or learning of the answer; fail-closed on missing/empty documents.
        read_docs: Dict[str, str] = {}
        try:
            for doc in researcher.get_research_sources() or []:
                url = str(doc.get("url") or "")
                if url:
                    read_docs[url] = str(doc.get("raw_content") or doc.get("content") or "")
        except (AttributeError, TypeError):
            read_docs = {}
        self._read_docs.update(read_docs)
        claims = [s for s in re.split(r"(?<=[.!?])\s+", node.answer_md) if s.strip()]
        claims += node.learnings
        node.sources = [u for u in node.sources
                        if read_docs.get(u)
                        and any(text_supported(c, read_docs[u]) for c in claims)]

        node.tokens_spent = (len(context) + len(response)) // 4
        try:
            node.credits_spent = float(researcher.get_costs() or 0.0)
        except (AttributeError, TypeError, ValueError):
            node.credits_spent = 0.0
        node.status = NodeStatus.ANSWERED

    async def generate_child_questions(self, node: ResearchNode) -> List[str]:
        """Self-Ask expansion: follow-up questions from the node's answer + gaps."""
        existing = [n.question for n in self.nodes.values()][:30]
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher generating disjoint follow-up research questions."},
                {"role": "user",
                 "content": (
                     f"Researched question: {node.question}\n"
                     f"Answer digest: {node.answer_digest}\n\n"
                     f"Existing questions in this research tree (do NOT overlap them):\n"
                     + "\n".join(f"- {q}" for q in existing)
                     + f"\n\nGenerate up to {self._max_breadth} follow-up questions that fill gaps "
                       "left by the answer. Each must be disjoint from the others and from the "
                       "existing questions. Return 0 questions if the answer is complete. "
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

    def compute_novelty(self, node: ResearchNode) -> float:
        """Fraction of the node's learnings not already seen elsewhere in the tree."""
        items = node.learnings or ([node.answer_digest] if node.answer_digest else [])
        if not items:
            return 1.0
        keys = {" ".join(str(l).lower().split()) for l in items}
        new = [k for k in keys if k not in self._seen_learnings]
        self._seen_learnings.update(keys)
        return len(new) / len(keys)

    async def synthesize_node(self, node: ResearchNode,
                              child_summaries: Optional[List[str]] = None,
                              citation_ids: Optional[List[str]] = None) -> str:
        """Roll one node up into a summary (leaf: digest; internal: LLM roll-up)."""
        child_summaries = [s for s in (child_summaries or []) if s]
        cites = " ".join(f"[{c}]" for c in (citation_ids or []))
        if node.status == NodeStatus.PENDING:
            return f"(unexplored frontier) {node.question}"
        if not child_summaries:
            base = node.answer_digest or node.answer_md or node.question
            return f"{base} {cites}".strip()
        blocks = "\n\n".join(f"### Sub-finding\n{s}" for s in child_summaries)
        response = await create_chat_completion(
            messages=[
                {"role": "system",
                 "content": "You are an expert researcher synthesizing findings into a coherent markdown section."},
                {"role": "user",
                 "content": (
                     f"Question: {node.question}\n"
                     f"Own findings: {node.answer_digest} {cites}\n\n"
                     f"Sub-findings from follow-up research:\n{blocks}\n\n"
                     "Synthesize everything into one coherent markdown answer to the question. "
                     "Keep existing [id] citation markers attached to the claims they support."
                 )},
            ],
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.3,
        )
        return str(response or "").strip()

    # -------------------------------------------------------------------- run

    async def run(self, query: Optional[str] = None, max_depth: int = 3,
                  max_breadth: int = 4, max_nodes: int = 40,
                  token_budget: int = 300_000, credit_budget: float = 150.0,
                  novelty_threshold: float = 0.30, expansion_policy: str = "best_first",
                  stream: bool = False, outputs_dir: Optional[str] = None,
                  time_budget_s: float = 600.0) -> Dict[str, Any]:
        query = query or self.researcher.query
        self._max_breadth = max_breadth
        start = time.time()

        root = ResearchNode(id="0", question=query, parent_id=None, depth=0)
        root.priority = 1.0
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
            ids = [url_to_id[u] for u in n.sources if u in url_to_id]
            text = await self.synthesize_node(n, summaries, ids)
            self._syntheses[n.id] = text
            return text

        root_text = "" if root.status == NodeStatus.PRUNED else await rollup(root)

        lines = [f"# {query}", "", root_text or "_(no synthesis)_", ""]
        if citation_map:
            lines += ["## Citations", ""]
            lines += [f"- [{cid}] {url}" for cid, url in citation_map.items()]
        report_md = "\n".join(lines).strip() + "\n"

        # defect 6a fail-closed: an [id] with no citations entry never reaches
        # the caller — detect, then strip the unbacked markers from the report.
        uncited_ids = find_uncited_ids(report_md, citation_map)
        if uncited_ids:
            logger.error(f"uncited [id] markers stripped from report: {uncited_ids}")
            report_md = re.sub(
                r"\[(?:" + "|".join(re.escape(c) for c in uncited_ids) + r")\]",
                "", report_md)

        # CitationAgent over the tree node claims: each learning is checked
        # against its node's read-and-quoted document (no re-scrape needed).
        claim_urls: Dict[str, str] = {}
        for n in self.nodes.values():
            url = n.sources[0] if n.sources else ""
            for learning in n.learnings:
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
        return {
            "id": n.id,
            "question": n.question,
            "status": n.status.value,
            "children": list(n.children),
            "parent_id": n.parent_id,
            "depth": n.depth,
            "sources": list(n.sources),
            "novelty": n.novelty,
            "priority": n.priority,
            "answer_digest": n.answer_digest,
        }

    def _persist(self, outputs_dir: str, query: str, meta: Dict[str, Any],
                 citation_map: Dict[str, str], report_md: str) -> Dict[str, str]:
        """Write tree.json (smoke-gate contract shape) + final report markdown."""
        out = Path(outputs_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{_slug(query)}-{uuid.uuid4().hex[:8]}"
        payload = {
            "meta": meta,
            "nodes": [{"id": n.id, "depth": n.depth, "status": n.status.value,
                       "question": n.question} for n in self.nodes.values()],
            "citations": citation_map,
        }
        tree_path = out / f"{stem}.tree.json"
        report_path = out / f"{stem}.tree-report.md"
        tree_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        report_path.write_text(report_md, encoding="utf-8")
        return {"tree_json": str(tree_path), "report_md": str(report_path)}
