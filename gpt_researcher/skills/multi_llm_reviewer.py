"""Multi-LLM Consensus Review skill for GPT Researcher.

Orchestrates parallel reviews of research context by multiple LLM providers
(e.g., Gemini, ChatGPT, Claude), merges their feedback, and conducts
supplementary research based on identified gaps.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..actions.utils import stream_output
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


REVIEWER_SYSTEM_PROMPT = (
    "You are an expert research reviewer. Your task is to critically evaluate "
    "research context gathered for a specific query. You must identify gaps, "
    "suggest additional research queries, and provide constructive criticism. "
    "Be specific and actionable.\n\n"
    "You MUST respond with a JSON object in this exact format:\n"
    '{"gaps": ["gap1", "gap2"], "additional_queries": ["query1", "query2"], '
    '"critiques": ["critique1", "critique2"], "confidence": 0.7, '
    '"strengths": ["strength1", "strength2"]}\n\n'
    "IMPORTANT: Return ONLY the JSON object, no additional text."
)

MERGE_SYSTEM_PROMPT = (
    "You are an expert at synthesizing multiple review perspectives into "
    "actionable research feedback. Identify consensus points (mentioned by 2+ "
    "reviewers) and unique insights. Prioritize queries by potential research value.\n\n"
    "You MUST respond with a JSON object in this exact format:\n"
    '{"consensus_gaps": ["gap mentioned by 2+ reviewers"], '
    '"unique_gaps": ["gap mentioned by only 1 reviewer"], '
    '"prioritized_queries": ["most valuable query first"], '
    '"consensus_critiques": ["critique agreed by majority"], '
    '"key_strengths": ["strength noted by reviewers"]}\n\n'
    "IMPORTANT: Return ONLY the JSON object, no additional text."
)


@dataclass
class ReviewResult:
    """Result from a single LLM reviewer."""
    reviewer_id: str
    provider: str
    model: str
    gaps: List[str] = field(default_factory=list)
    additional_queries: List[str] = field(default_factory=list)
    critiques: List[str] = field(default_factory=list)
    confidence: float = 0.5
    strengths: List[str] = field(default_factory=list)
    cost: float = 0.0
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class MergedReview:
    """Merged review from all LLM reviewers."""
    consensus_gaps: List[str] = field(default_factory=list)
    unique_gaps: List[str] = field(default_factory=list)
    prioritized_queries: List[str] = field(default_factory=list)
    consensus_critiques: List[str] = field(default_factory=list)
    key_strengths: List[str] = field(default_factory=list)
    reviewer_count: int = 0
    total_cost: float = 0.0
    supplementary_context: str = ""


class MultiLLMReviewer:
    """Orchestrates multi-LLM review of research context.

    Sends research context to multiple LLM providers in parallel,
    collects independent reviews, merges feedback, and optionally
    conducts supplementary research based on identified gaps.
    """

    def __init__(self, researcher):
        self.researcher = researcher
        self.cfg = researcher.cfg
        self.websocket = researcher.websocket
        self._review_costs = {}

    def is_enabled(self) -> bool:
        return getattr(self.cfg, 'multi_llm_review_enabled', False)

    def _get_reviewer_configs(self) -> List[Dict[str, str]]:
        models_config = getattr(self.cfg, 'multi_llm_review_models', None)
        if not models_config:
            models_config = [
                "google_genai:gemini-3.1-pro",
                "openai:gpt-5.5",
                "anthropic:claude-opus-4-7",
            ]

        from ..config.config import Config
        reviewers = []
        for model_str in models_config:
            try:
                provider, model = Config.parse_llm(model_str)
                reviewers.append({
                    "provider": provider,
                    "model": model,
                    "full_name": model_str,
                })
            except Exception as e:
                logger.warning(f"Skipping invalid reviewer model '{model_str}': {e}")
        return reviewers

    async def review_context(
        self,
        query: str,
        context: Any,
        max_additional_queries: int = 3,
    ) -> MergedReview:
        """Run multi-LLM review on research context."""
        if self.researcher.verbose:
            await stream_output(
                "logs", "multi_llm_review_start",
                f"\n🔍 Starting multi-LLM consensus review with multiple perspectives...",
                self.websocket,
            )

        reviewers = self._get_reviewer_configs()
        if not reviewers:
            logger.error("No valid reviewer configurations found")
            return MergedReview()

        context_str = self._prepare_context(context)

        # Phase 1: Parallel independent reviews
        review_results = await self._run_parallel_reviews(query, context_str, reviewers)
        successful_reviews = [r for r in review_results if r.error is None]

        if not successful_reviews:
            logger.error("All LLM reviews failed")
            if self.researcher.verbose:
                await stream_output(
                    "logs", "multi_llm_review_failed",
                    "❌ All LLM reviewers failed. Proceeding without review.",
                    self.websocket,
                )
            return MergedReview()

        if self.researcher.verbose:
            reviewer_names = [r.reviewer_id for r in successful_reviews]
            await stream_output(
                "logs", "multi_llm_review_phase1_done",
                f"✅ Received {len(successful_reviews)}/{len(reviewers)} reviews "
                f"from: {', '.join(reviewer_names)}. Merging feedback...",
                self.websocket,
            )

        # Phase 2: Merge reviews
        merged = await self._merge_reviews(query, successful_reviews)

        # Phase 3: Supplementary research
        do_supplementary = getattr(self.cfg, 'multi_llm_review_supplementary', True)
        if do_supplementary and merged.prioritized_queries:
            queries_to_run = merged.prioritized_queries[:max_additional_queries]
            if self.researcher.verbose:
                await stream_output(
                    "logs", "multi_llm_review_supplementary",
                    f"🔎 Running {len(queries_to_run)} supplementary queries based on review gaps...",
                    self.websocket,
                )
            supplementary = await self._run_supplementary_research(queries_to_run)
            merged.supplementary_context = supplementary

        # Final summary
        total_cost = sum(self._review_costs.values())
        merged.total_cost = total_cost
        if self.researcher.verbose:
            await stream_output(
                "logs", "multi_llm_review_complete",
                f"✅ Multi-LLM review complete. {len(successful_reviews)} reviewers, "
                f"{len(merged.consensus_gaps)} consensus gaps, "
                f"{len(merged.prioritized_queries)} follow-up queries. "
                f"Review cost: ${total_cost:.4f}",
                self.websocket,
            )

        return merged

    def _prepare_context(self, context: Any) -> str:
        if isinstance(context, list):
            return "\n\n".join(str(c) for c in context)
        return str(context)

    async def _run_parallel_reviews(
        self,
        query: str,
        context_str: str,
        reviewers: List[Dict[str, str]],
    ) -> List[ReviewResult]:
        tasks = [
            self._single_review(query, context_str, reviewer, i)
            for i, reviewer in enumerate(reviewers)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        review_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                reviewer = reviewers[i]
                logger.error(f"Review failed for {reviewer['full_name']}: {result}")
                review_results.append(ReviewResult(
                    reviewer_id=reviewer["full_name"],
                    provider=reviewer["provider"],
                    model=reviewer["model"],
                    error=str(result),
                ))
            else:
                review_results.append(result)

        return review_results

    async def _single_review(
        self,
        query: str,
        context_str: str,
        reviewer: Dict[str, str],
        index: int,
    ) -> ReviewResult:
        start_time = time.time()
        cost_tracker = {"cost": 0.0}

        def cost_cb(cost: float):
            cost_tracker["cost"] += cost

        prompt = self.researcher.prompt_family.generate_multi_llm_review_prompt(
            query=query, context=context_str,
        )

        try:
            response = await create_chat_completion(
                model=reviewer["model"],
                messages=[
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                llm_provider=reviewer["provider"],
                max_tokens=getattr(self.cfg, 'multi_llm_review_token_limit', 2000),
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=cost_cb,
            )

            parsed = self._parse_review_response(response)
            duration = time.time() - start_time
            self._review_costs[f"reviewer_{index}"] = cost_tracker["cost"]

            return ReviewResult(
                reviewer_id=reviewer["full_name"],
                provider=reviewer["provider"],
                model=reviewer["model"],
                gaps=parsed.get("gaps", []),
                additional_queries=parsed.get("additional_queries", []),
                critiques=parsed.get("critiques", []),
                confidence=parsed.get("confidence", 0.5),
                strengths=parsed.get("strengths", []),
                cost=cost_tracker["cost"],
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Review error ({reviewer['full_name']}): {e}")
            return ReviewResult(
                reviewer_id=reviewer["full_name"],
                provider=reviewer["provider"],
                model=reviewer["model"],
                cost=cost_tracker["cost"],
                error=str(e),
                duration_seconds=duration,
            )

    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        # Gemini (langchain_google_genai) may deliver content as list[ContentPart]
        # instead of str; json.loads/re/slicing below assume str. A list raises
        # TypeError ("the JSON object must be str ... not list") which the
        # JSONDecodeError handlers do NOT catch -> review crashes. Coerce first.
        if not isinstance(response, str):
            if isinstance(response, list):
                response = "".join(
                    part if isinstance(part, str)
                    else str(part.get("text") or part.get("content") or "")
                    if isinstance(part, dict) else str(part)
                    for part in response if part is not None
                )
            else:
                response = "" if response is None else str(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            import json_repair
            return json_repair.loads(response)
        except Exception:
            pass

        logger.warning("Could not parse review response as JSON, using fallback")
        return {
            "gaps": [],
            "additional_queries": [],
            "critiques": [response[:500]] if response else [],
            "confidence": 0.3,
            "strengths": [],
        }

    async def _merge_reviews(
        self,
        query: str,
        reviews: List[ReviewResult],
    ) -> MergedReview:
        if len(reviews) == 1:
            r = reviews[0]
            return MergedReview(
                consensus_gaps=r.gaps,
                unique_gaps=[],
                prioritized_queries=r.additional_queries,
                consensus_critiques=r.critiques,
                key_strengths=r.strengths,
                reviewer_count=1,
            )

        cost_tracker = {"cost": 0.0}
        def cost_cb(cost: float):
            cost_tracker["cost"] += cost

        reviews_summary = self._format_reviews_for_merge(reviews)
        prompt = self.researcher.prompt_family.generate_review_merge_prompt(
            query=query, reviews=reviews_summary,
        )

        try:
            response = await create_chat_completion(
                model=self.cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                llm_provider=self.cfg.smart_llm_provider,
                max_tokens=getattr(self.cfg, 'multi_llm_review_token_limit', 2000),
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=cost_cb,
            )

            self._review_costs["merge"] = cost_tracker["cost"]
            parsed = self._parse_review_response(response)

            return MergedReview(
                consensus_gaps=parsed.get("consensus_gaps", []),
                unique_gaps=parsed.get("unique_gaps", []),
                prioritized_queries=parsed.get("prioritized_queries", []),
                consensus_critiques=parsed.get("consensus_critiques", []),
                key_strengths=parsed.get("key_strengths", []),
                reviewer_count=len(reviews),
            )
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return self._manual_merge(reviews)

    def _format_reviews_for_merge(self, reviews: List[ReviewResult]) -> str:
        parts = []
        for i, r in enumerate(reviews, 1):
            parts.append(
                f"=== Reviewer {i} ({r.reviewer_id}) ===\n"
                f"Confidence: {r.confidence}\n"
                f"Gaps: {json.dumps(r.gaps)}\n"
                f"Additional Queries: {json.dumps(r.additional_queries)}\n"
                f"Critiques: {json.dumps(r.critiques)}\n"
                f"Strengths: {json.dumps(r.strengths)}"
            )
        return "\n\n".join(parts)

    def _manual_merge(self, reviews: List[ReviewResult]) -> MergedReview:
        all_gaps = []
        all_queries = []
        all_critiques = []
        all_strengths = []

        for r in reviews:
            all_gaps.extend(r.gaps)
            all_queries.extend(r.additional_queries)
            all_critiques.extend(r.critiques)
            all_strengths.extend(r.strengths)

        return MergedReview(
            consensus_gaps=list(set(all_gaps)),
            unique_gaps=[],
            prioritized_queries=list(dict.fromkeys(all_queries)),
            consensus_critiques=list(set(all_critiques)),
            key_strengths=list(set(all_strengths)),
            reviewer_count=len(reviews),
        )

    async def _run_supplementary_research(self, queries: List[str]) -> str:
        conductor = self.researcher.research_conductor
        supplementary_contexts = []

        for query in queries:
            try:
                if self.researcher.verbose:
                    await stream_output(
                        "logs", "supplementary_research",
                        f"  📖 Supplementary research: '{query}'",
                        self.websocket,
                    )
                context = await conductor._process_sub_query(
                    query,
                    scraped_data=[],
                    query_domains=self.researcher.query_domains
                    if hasattr(self.researcher, 'query_domains') else [],
                )
                if context:
                    supplementary_contexts.append(str(context))
            except Exception as e:
                logger.error(f"Supplementary research failed for '{query}': {e}")

        return "\n\n".join(supplementary_contexts)
