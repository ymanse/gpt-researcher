from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import time
from datetime import datetime, timedelta

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource
from ..actions.query_processing import get_search_results
from ..actions.agent_creator import choose_agent

logger = logging.getLogger(__name__)

# Maximum words allowed in context (25k words for safety margin)
MAX_CONTEXT_WORDS = 25000

# Appended to the context of a run that stopped expanding on a spent CLI-session
# allowance. A partial answer presented as a complete one is worse than the
# AgentBudgetExceeded it replaces: the reader cannot tell, and neither can the report
# writer downstream.
BUDGET_TRUNCATION_NOTICE = """

## Incomplete Research
This research stopped early: the run's CLI-session budget was spent before every
planned query and follow-up depth had been researched. The findings above are PARTIAL
— questions that were planned but never investigated are missing entirely, and nothing
above should be read as a complete answer to the original query.
"""


def count_words(text) -> int:
    """Count words in a text string. Handles both strings and lists."""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return len(str(text).split())

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit while preserving most recent/relevant items"""
    total_words = 0
    trimmed_context = []

    # Process in reverse to keep most recent items
    for item in reversed(context_list):
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        else:
            break

    return trimmed_context

class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # Start from 1 and increment up to total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # Start from 0 and count up to total_breadth as queries complete
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # Track all research sources
        self.context = []  # Track all context
        self.scope_brief = None  # Set by run(scope=True): {"query", "questions", "scope_statement"}
        # Resolved ONCE per run by _resolve_run_context(), then handed to every nested
        # researcher. Measured 2026-09-09 on a breadth=2/depth=2 run: without them the
        # 6 nested researchers pay 6 choose_agent sessions and 12 routing
        # classifications, against a pooled ceiling that was 53 that day.
        self._category = None
        self._agent = None
        self._role = None
        self._run_context_resolved = False
        # True once the CLI-session allowance stopped this run expanding. Travels out
        # in deep_research()'s result and as a disclosure in the context run() returns.
        self.budget_exhausted = False
        # Whether ANY sub-query has produced research this run. The budget gate
        # reads it so a run that has gathered nothing never reports itself as
        # "partial" — see process_query.
        self._researched_any = False

    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[Dict[str, str]]:
        """Generate SERP queries for research"""
        messages = [
            {"role": "system", "content": "You are an expert researcher generating search queries."},
            {"role": "user",
             "content": f"Given the following prompt, generate {num_queries} unique search queries to research the topic thoroughly. For each query, provide a research goal. Format as 'Query: <query>' followed by 'Goal: <goal>' for each pair: {query}"}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=self.researcher.cfg.reasoning_effort,
            temperature=0.4
        )

        lines = response.split('\n')
        queries = []
        current_query = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Query:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query': line.replace('Query:', '').strip()}
            elif line.startswith('Goal:') and current_query:
                current_query['researchGoal'] = line.replace('Goal:', '').strip()

        if current_query:
            queries.append(current_query)

        return queries[:num_queries]

    async def generate_research_plan(self, query: str, num_questions: int = 3) -> List[str]:
        """Generate follow-up questions to clarify research direction"""
        # Get initial search results from all retrievers to inform query generation
        all_search_results = []
        for retriever in self.researcher.retrievers:
            try:
                results = await get_search_results(
                    query,
                    retriever,
                    researcher=self.researcher
                )
                all_search_results.extend(results)
            except Exception as e:
                logger.warning(f"Error with retriever {retriever.__name__}: {e}")
        search_results = all_search_results
        logger.info(f"Initial web knowledge obtained: {len(search_results)} results")

        # Get current time for context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = [
            {"role": "system", "content": "You are an expert researcher. Your task is to analyze the original query and search results, then generate targeted questions that explore different aspects and time periods of the topic."},
            {"role": "user",
             "content": f"""Original query: {query}

Current time: {current_time}

Search results:
{search_results}

Based on these results, the original query, and the current time, generate {num_questions} unique questions. Each question should explore a different aspect or time period of the topic, considering recent developments up to {current_time}.

Format each question on a new line starting with 'Question: '"""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4
        )

        questions = [q.replace('Question:', '').strip()
                     for q in response.split('\n')
                     if q.strip().startswith('Question:')]
        return questions[:num_questions]

    async def process_research_results(self, query: str, context: str, num_learnings: Optional[int] = None) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        if num_learnings is None:
            num_learnings = getattr(self.researcher.cfg, 'deep_research_learnings', 8)
        messages = [
            {"role": "system", "content": "You are an expert researcher analyzing search results."},
            {"role": "user",
             "content": f"Given the following research results for the query '{query}', extract the {num_learnings} most important key learnings and suggest follow-up questions. For each learning, include a citation to the source URL if available. Format each learning as 'Learning [source_url]: <insight>' and each question as 'Question: <question>':\n\n{context}"}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
            reasoning_effort=ReasoningEfforts.High.value,
            max_tokens=getattr(self.researcher.cfg, 'deep_research_learnings_tokens', 2500)
        )

        lines = response.split('\n')
        learnings = []
        questions = []
        citations = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Learning'):
                import re
                url_match = re.search(r'\[(.*?)\]:', line)
                if url_match:
                    url = url_match.group(1)
                    # split after the "[url]:" marker — split(':', 1) would cut at
                    # the "https:" inside the brackets and corrupt the learning text
                    learning = line[url_match.end():].strip()
                    learnings.append(learning)
                    citations[learning] = url
                else:
                    # Try to find URL in the line itself
                    url_match = re.search(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                    if url_match:
                        url = url_match.group(0)
                        learning = line.replace(url, '').replace('Learning:', '').strip()
                        learnings.append(learning)
                        citations[learning] = url
                    else:
                        learnings.append(line.replace('Learning:', '').strip())
            elif line.startswith('Question:'):
                questions.append(line.replace('Question:', '').strip())

        learnings = learnings[:num_learnings]
        print(f"TIERA_EVIDENCE stage=2 learnings_count={len(learnings)}", flush=True)
        return {
            'learnings': learnings,
            'followUpQuestions': questions[:num_learnings],
            'citations': citations
        }

    def scraped_documents(self) -> Dict[str, str]:
        """url -> page text for every source this run already read.

        Seeds the citation pass so it checks quotes against pages we still hold in
        memory. Without it every claim re-fetches its source through firecrawl with
        maxAge=0 (cache bypass), serially: measured 13 minutes of a 37-minute run
        re-scraping 66 urls the research had scraped minutes earlier.
        """
        documents: Dict[str, str] = {}
        # Read the store directly rather than through get_research_sources(): the
        # seed is an optimisation, so a researcher that doesn't track sources should
        # degrade to fetching, not raise.
        for source in getattr(self.researcher, 'research_sources', None) or []:
            url = source.get('url')
            content = source.get('raw_content') or source.get('content')
            # Empty content must NOT be seeded: an entry for a url is a claim that
            # we have the page, and an empty one would mark its claims unverified
            # rather than falling through to a fetch.
            if url and content:
                documents[url] = content
        return documents

    async def verify_citations(self, citations: Dict[str, str]) -> Dict[str, Any]:
        """Verify collected claims against their cited sources (stage 3)."""
        from .citation_verification import CitationAgent
        return await asyncio.to_thread(
            CitationAgent().verify, citations, self.scraped_documents()
        )

    async def _resolve_run_context(self, query: str) -> None:
        """Decide once, for the whole run, what every nested researcher re-decides.

        Two CLI sessions here replace one choose_agent per nested researcher plus one
        routing classification per sub-query of each of them.

        The classification half is a REGRESSION, not an old cost: until 2026-09-06 the
        retriever was constructed without the researcher, so SmartRetriever.cfg was
        None and _classify_query answered "general_web" on its very first branch for
        ZERO sessions. Passing the researcher fixed smart routing — a real bug — and
        switched those sessions on. The tree caps them at one per run by forcing the
        category; until now the linear path had no such cap.

        Best-effort, both halves: a research run must not die because a routing hint or
        a role prompt could not be produced. Whichever one fails falls back to today's
        per-researcher behaviour.
        """
        # The original question, not the follow-up blob run() assembles around it
        # ("Q: ... / A: Automatically proceeding with research" x3): this decides the
        # retriever bundle and the role prompt for the whole run, and the boilerplate is
        # not what the run is about. Same input the tree resolves against.
        query = str(getattr(self.researcher, 'query', '') or query)
        if self._run_context_resolved:
            return
        # Set BEFORE resolving: the recursion re-enters deep_research once per depth
        # level, and a failed resolution must be paid for at most once per run.
        self._run_context_resolved = True

        # Through the real classifier and not a constant: the category selects which
        # retriever bundle every sub-query searches with, so a hardcoded value would
        # silently change what the run reads. One call per run fits in every budget.
        try:
            from ..retrievers.smart.smart_retriever import SmartRetriever

            category = await asyncio.to_thread(
                SmartRetriever(query, cfg=self.researcher.cfg)._classify_query)
            self._category = str(category) if category else None
            logger.info("deep research routing category resolved once: %s",
                        self._category)
        except Exception as e:
            # _classify_query swallows its own errors and answers "general_web", so
            # reaching here means the construction failed, not the classification.
            logger.warning("run-level classification failed, every sub-query classifies "
                           "itself as before: %s", e)

        # conduct_research's guard is an AND (`if not (agent and role)`), so a
        # half-resolved pair buys nothing: keep both or neither.
        try:
            agent, role = await choose_agent(
                query=query,
                cfg=self.researcher.cfg,
                parent_query=getattr(self.researcher, 'parent_query', '') or '',
                cost_callback=getattr(self.researcher, 'add_costs', None),
                headers=self.headers,
                prompt_family=getattr(self.researcher, 'prompt_family', None),
            )
            if agent and role:
                self._agent, self._role = agent, role
                logger.info("deep research agent resolved once: %s", agent)
        except Exception as e:
            logger.warning("run-level agent selection failed, every sub-query chooses "
                           "its own as before: %s", e)

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        print(f"\n📊 DEEP RESEARCH: depth={depth}, breadth={breadth}, query={query[:100]}...", flush=True)
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        # Idempotent, so the recursion below pays nothing for it.
        await self._resolve_run_context(query)

        # The linear path has no way to come back shallower, so a spent allowance used
        # to surface as AgentBudgetExceeded from whichever LLM call happened to be
        # next — and generate_search_queries, the first thing the recursion does, sits
        # outside every `except`. Measured 2026-09-09: the whole call died as "Failed
        # to get response from claude_agent API", 22 error rounds, no partial answer,
        # while a tree run four minutes earlier shared the same pooled ceiling of 53.
        # So the budget is tested BEFORE further work is started instead, and what has
        # already been gathered is kept. The reserve leaves the write-up solvent, the
        # same way the tree's expansion loop holds calls back for its roll-up.
        #
        # ponytail: the run's OWN first query is not gated. There is no partial result
        # to protect before any query has run, and an empty context handed to the report
        # writer is worse than an exception — it gets written up from prior knowledge.
        #
        # Imported here rather than at module scope: agent.py imports this module while
        # the package is initialising, and claude_agent/__init__ pulls in
        # claude_agent_sdk (0.27s, and an ImportError in any environment that installed
        # the fork without it). generic/base.py already loads that provider on demand
        # behind _check_pkg; a budget read must not be the thing that makes it eager.
        from ..llm_provider.claude_agent._subscription import (
            agent_budget_exhausted,
            agent_synthesis_reserve,
        )

        synthesis_reserve = agent_synthesis_reserve()

        # Generate search queries
        print(f"🔎 Generating {breadth} search queries...", flush=True)
        serp_queries = await self.generate_search_queries(query, num_queries=breadth)
        print(f"✅ Generated {len(serp_queries)} queries: {[q['query'] for q in serp_queries]}", flush=True)
        progress.total_queries = len(serp_queries)

        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        all_context = []
        all_sources = []

        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                # Checked inside the semaphore, so a query queued behind the ones that
                # spent the allowance sees the spend rather than the state at gather().
                #
                # `self._researched_any` is the half that was missing, and its absence
                # was worse than the exception it replaced. Allowances POOL, so a
                # concurrent run can leave this one exhausted before its FIRST sub-query:
                # gating unconditionally then researched nothing, appended the
                # "Incomplete Research" banner to an EMPTY context, and handed that to
                # the report writer — which writes a confident-looking report out of
                # prior knowledge. Measured over allowances 3/5/6/7: 0 of 6 researchers
                # ran, 0 learnings, 0 context items, truncated=True every time.
                #
                # Degrading is only honest when there is something to degrade TO. With
                # nothing gathered yet, let the query run and let a spent budget raise:
                # a clear failure beats a fabricated answer.
                if self._researched_any and agent_budget_exhausted(reserve=synthesis_reserve):
                    self.budget_exhausted = True
                    logger.warning(
                        "CLI-session allowance spent: sub-query %r is left "
                        "unresearched and the research already done is kept",
                        serp_query['query'])
                    return None
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=ReportSource.Web.value,
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls,
                        # Both, or conduct_research's guard (`if not (agent and role)`)
                        # does not fire and this researcher re-chooses a role prompt the
                        # run already picked. None for either is exactly today's cost.
                        agent=self._agent,
                        role=self._role,
                        # NOT preset_sub_queries, and that is deliberate. A tree node IS
                        # one question, so presetting costs it nothing; a linear
                        # sub-query is not — this researcher's own planning is the
                        # SECOND level of decomposition, and it is what linear `depth`
                        # is made of. Presetting it would quietly redefine depth=2.
                        # Propagate MCP configuration to nested researchers
                        mcp_configs=self.researcher.mcp_configs,
                        mcp_strategy=self.researcher.mcp_strategy
                    )
                    # The routing category is a property of the RUN, not of the
                    # sub-query: _classify_query short-circuits on it without asking the
                    # FAST_LLM. Stamped after construction because Config builds its own
                    # cfg instance; an unresolved category leaves it None, which is what
                    # default.py ships and means "classify normally".
                    if self._category:
                        try:
                            researcher.cfg.smart_retriever_force_category = self._category
                        # a cfg that refuses the attribute is not worth failing a
                        # research run over: the sub-query classifies itself instead
                        except AttributeError:
                            logger.warning(
                                "could not stamp the run category onto the researcher "
                                "for %r", serp_query['query'])

                    # Conduct research
                    context = await researcher.conduct_research()

                    # Get results and visited URLs
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # Process results to extract learnings and citations
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # Update progress
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': "\n".join(context) if isinstance(context, list) else (context or ""),
                        'sources': sources if sources else []
                    }

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    print(f"\n❌ DEEP RESEARCH ERROR: {str(e)}\n{error_details}", flush=True)
                    return None

        # Process queries concurrently with limit
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Update breadth progress based on successful queries
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # Collect all results
        for result in results:
            all_learnings.extend(result['learnings'])
            # This run has something to degrade to from here on.
            self._researched_any = True
            all_visited_urls.update(result['visited_urls'])
            all_citations.update(result['citations'])
            if result['context']:
                # Use extend, not append: when CURATE_SOURCES=True, result['context'] is
                # a List[dict]. append() nests it as a single item, which causes
                # "\n".join() to crash later with "expected str instance, dict found".
                ctx = result['context']
                if isinstance(ctx, list):
                    all_context.extend(ctx)
                else:
                    all_context.append(ctx)
            if result['sources']:
                all_sources.extend(result['sources'])

            # Continue deeper if needed
            if depth > 1:
                # `continue`, not `break`: the shallow results of the remaining
                # queries are already gathered above and must still be collected.
                if agent_budget_exhausted(reserve=synthesis_reserve):
                    self.budget_exhausted = True
                    logger.warning(
                        "CLI-session allowance spent: not recursing to depth %s. The "
                        "learnings gathered so far are kept and reported as partial.",
                        depth - 1)
                    continue
                new_breadth = max(2, breadth // 2)
                new_depth = depth - 1
                progress.current_depth += 1

                # Create next query from research goal and follow-up questions
                next_query = f"""
                Previous research goal: {result['researchGoal']}
                Follow-up questions: {' '.join(result['followUpQuestions'])}
                """

                # Recursive research
                deeper_results = await self.deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    citations=all_citations,
                    visited_urls=all_visited_urls,
                    on_progress=on_progress
                )

                all_learnings = deeper_results['learnings']
                all_visited_urls.update(deeper_results['visited_urls'])
                all_citations.update(deeper_results['citations'])
                if deeper_results.get('context'):
                    all_context.extend(deeper_results['context'])
                if deeper_results.get('sources'):
                    all_sources.extend(deeper_results['sources'])

        # Update class tracking
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # Trim context to stay within word limits
        trimmed_context = trim_context_to_word_limit(all_context)
        logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources,
            # A caller cannot otherwise tell a budget-truncated result from a complete
            # one — which is worse than the exception this replaces.
            'agent_budget_exhausted': self.budget_exhausted,
        }

    async def run(self, on_progress=None, scope: bool = False) -> str:
        """Run the deep research process and generate final report"""
        print(f"\n🔍 DEEP RESEARCH: Starting with breadth={self.breadth}, depth={self.depth}, concurrency={self.concurrency_limit}", flush=True)
        start_time = time.time()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        follow_up_questions = await self.generate_research_plan(self.researcher.query)

        if scope:
            # Stage 5: 1-round scope brief — resolve the clarification questions with
            # an LLM scope statement instead of the auto-answer boilerplate.
            questions_block = "\n".join(f"- {q}" for q in follow_up_questions)
            scope_statement = await create_chat_completion(
                messages=[
                    {"role": "system",
                     "content": "You are an expert researcher. Resolve clarification questions into one concise scope statement."},
                    {"role": "user",
                     "content": f"Original query: {self.researcher.query}\n\nClarification questions:\n{questions_block}\n\nAnswer them with the most reasonable defaults and write a single concise scope statement pinning down the research scope."},
                ],
                llm_provider=self.researcher.cfg.strategic_llm_provider,
                model=self.researcher.cfg.strategic_llm_model,
                temperature=0.2,
            )
            self.scope_brief = {
                "query": self.researcher.query,
                "questions": list(follow_up_questions),
                "scope_statement": scope_statement,
            }
            brief_text = f"{questions_block}\n{scope_statement}"
            print(f"TIERA_EVIDENCE stage=5 brief_present=1 brief_len={len(brief_text)}", flush=True)
            combined_query = f"""
        Initial Query: {self.researcher.query}\nConfirmed Scope:\n{scope_statement}
        """
        else:
            answers = ["Automatically proceeding with research"] * len(follow_up_questions)

            qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
            combined_query = f"""
        Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        """ + "\n".join(qa_pairs)

        results = await self.deep_research(
            query=combined_query,
            breadth=self.breadth,
            depth=self.depth,
            on_progress=on_progress
        )

        # BEFORE verification, not after it. scraped_documents() reads
        # researcher.research_sources to seed the citation pass with pages this run
        # already holds; that assignment used to sit ~45 lines below, so at this point
        # the seed was empty and the optimisation never fired — every claim re-fetched
        # through firecrawl with maxAge=0, which is the 13-minutes-of-37 the seed was
        # added to remove. Worse since `unretrieved` exists: a re-fetch that fails then
        # marks a page we are holding in memory as one we could not obtain, inventing
        # the very evidence-hole that flag is meant to report honestly.
        if results.get('sources'):
            self.researcher.research_sources = results['sources']

        # Citation-verification post-pass over collected claims (stage 3)
        verification = await self.verify_citations(results['citations'])
        # unretrieved is reported SEPARATELY from unverified, not carved out of it.
        # Both numbers matter and they mean opposite things: unverified says the
        # evidence does not back the claim, unretrieved says we never got to look. A
        # line that prints only the first invites the reader to treat a fetch we lost
        # as a claim the sources refuted — which is how a correct finding was thrown
        # away. It stays a subset so the pinned arithmetic still holds.
        unretrieved = sum(1 for c in verification.get("claims", []) if c.get("unretrieved"))
        print(
            f"TIERA_EVIDENCE stage=3 total_claims={verification['total_claims']} "
            f"grounded={verification['grounded']} unverified={verification['unverified']} "
            f"unretrieved={unretrieved}",
            flush=True,
        )
        if unretrieved:
            logger.warning(
                f"{unretrieved} of {verification['unverified']} unverified claims cite a "
                f"source that was never obtained (blocked, rate-limited or unreachable). "
                f"That is a gap in the evidence, not evidence against the claim — do not "
                f"drop those claims on the strength of this pass."
            )

        # Get costs after deep research
        research_costs = self.researcher.get_costs() - initial_costs

        # Log research costs if we have a log handler
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # Prepare context with citations
        context_with_citations = []
        for learning in results['learnings']:
            citation = results['citations'].get(learning, '')
            if citation:
                context_with_citations.append(f"{learning} [Source: {citation}]")
            else:
                context_with_citations.append(learning)

        # Add all research context
        if results.get('context'):
            context_with_citations.extend(results['context'])

        # Trim final context to word limit
        final_context = trim_context_to_word_limit(context_with_citations)
        
        # Set enhanced context and visited URLs
        self.researcher.context = "\n".join(
            item if isinstance(item, str)
            else item.get("Content", str(item)) if isinstance(item, dict)
            else str(item)
            for item in final_context
        )
        # BUDGET_TRUNCATION_NOTICE goes into the context, not just the log: this
        # string is what the report is written from, so it is the only place a reader
        # can still be told. The linear equivalent of the tree's "## Unresearched
        # Questions" section — without it a run that stopped on a spent allowance is
        # indistinguishable from one that answered the question.
        # Only when there IS a partial result. An empty context with a "partial"
        # banner reads to the report writer as "research happened, here is some of it",
        # and it answers from prior knowledge instead of saying it has nothing. A run
        # that gathered nothing is a failure, not a shorter success.
        if results.get('agent_budget_exhausted') and self.researcher.context:
            logger.warning(
                "deep research stopped expanding on a spent CLI-session allowance; "
                "the context below is partial and says so.")
            self.researcher.context += BUDGET_TRUNCATION_NOTICE
        elif results.get('agent_budget_exhausted'):
            logger.error(
                "deep research gathered NO context before the CLI-session allowance "
                "was spent — reporting nothing rather than writing from prior knowledge")
        self.researcher.visited_urls = results['visited_urls']

        # (research_sources is assigned above, before verify_citations needs it)

        # Log total execution time
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")

        # Return the context - don't generate report here as it will be done by the main agent
        return self.researcher.context