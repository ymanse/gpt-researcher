"""Smart Retriever with LLM-based query classification and multi-retriever routing.

Routes search queries to optimal retriever combinations based on query type
(general web, code/technical, academic, news, comprehensive).
"""

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from ..utils import is_retryable_error

logger = logging.getLogger(__name__)

# Query categories and their retriever routing configurations.
# Each retriever entry: (retriever_name, max_results, extra_kwargs)
ROUTING_TABLE = {
    "general_web": [
        ("tavily", 7, {}),
        ("firecrawl", 5, {}),
        ("duckduckgo", 5, {}),
    ],
    "code_technical": [
        ("exa", 6, {"search_type": "neural"}),
        ("serper", 4, {"query_domains": ["github.com", "stackoverflow.com"]}),
        ("github", 5, {}),
    ],
    "academic": [
        ("firecrawl_research", 8, {"recency_days": 730}),
        ("arxiv", 5, {}),
        ("semantic_scholar", 5, {}),
    ],
    "news_current": [
        ("tavily", 8, {"topic": "news"}),
        ("serper", 4, {"time_range": "qdr:w"}),
        ("hackernews", 5, {"by_date": True}),
        ("bluesky", 5, {"sort": "latest"}),
        ("reddit", 4, {}),
        ("firecrawl", 5, {"tbs": "qdr:w"}),
    ],
    "comprehensive": [
        ("tavily", 3, {}),
        ("exa", 2, {"search_type": "neural"}),
        ("duckduckgo", 2, {}),
        ("arxiv", 2, {}),
        ("semantic_scholar", 2, {}),
        ("serper", 2, {}),
        ("hackernews", 2, {}),
        ("bluesky", 2, {}),
        ("reddit", 2, {}),
        ("github", 2, {}),
        ("firecrawl", 3, {}),
    ],
}

CLASSIFICATION_PROMPT = """Classify this search query into exactly ONE category. Reply with ONLY the category name, nothing else.

Categories:
- general_web: General facts, how-to, explanations, definitions
- code_technical: Programming, software, GitHub, APIs, debugging, libraries
- academic: Research papers, scientific theories, citations, studies
- news_current: Recent events, time-sensitive info, breaking news, stock prices
- comprehensive: Multi-domain complex queries requiring broad coverage

Query: {query}
Category:"""

# Prototype queries per category, for the embedding router below.
#
# The category DESCRIPTIONS in CLASSIFICATION_PROMPT are written for a reader who
# reasons; an encoder only measures topical proximity, and two of the five categories
# are not topical at all. `news_current` is about RECENCY and `comprehensive` is about
# the SHAPE of the question -- a description naming those properties embeds nowhere
# near a query that has them. Prototypes carry the property itself, so the vector for
# "comprehensive" sits where multi-clause survey questions actually land.
#
# Five per category, deliberately spread across sub-topics: the centroid of a narrow
# set is a point on one topic, and every query off that topic loses to whichever
# category happens to be broader.
CATEGORY_PROTOTYPES = {
    "general_web": [
        "how do I renew my passport",
        "what is the capital of Australia",
        "difference between weather and climate",
        "how does a refrigerator actually work",
        "best way to remove a coffee stain from a carpet",
    ],
    "code_technical": [
        "python asyncio gather exception handling",
        "react useEffect infinite loop fix",
        "how to configure nginx as a reverse proxy",
        "typescript generic constraints example",
        "git rebase versus merge workflow",
    ],
    "academic": [
        "transformer attention mechanism original paper",
        "meta-analysis of vitamin D supplementation trials",
        "CRISPR off-target effects systematic review",
        "empirical studies on reinforcement learning sample efficiency",
        "peer-reviewed research on microplastic toxicity",
    ],
    "news_current": [
        "latest AI regulation announcement this week",
        "current stock price of Nvidia",
        "who won the election yesterday",
        "breaking news on the earthquake",
        "recent developments in the trade negotiations",
    ],
    "comprehensive": [
        "compare the economic, environmental and social impacts of nuclear power",
        "full landscape of vector database vendors including pricing, benchmarks and adoption",
        "the history, technology and market of electric vehicles",
        "in-depth analysis of remote work effects on productivity, real estate and urban planning",
        "overview of quantum computing hardware approaches and their tradeoffs",
    ],
}

# How far ahead the winning category must be before the embedding router is trusted to
# decide on its own. Below it the query sits between two bundles and the FAST_LLM is
# asked, exactly as before -- the margin is what makes "no quality risk" a fact rather
# than a hope. Tuned on tests/search_quality/s12/classification_set.py; raise it to buy
# accuracy with sessions, lower it to buy sessions with accuracy, set 0 to never ask.
_EMBED_MARGIN = float(os.getenv("SMART_RETRIEVER_EMBED_MARGIN", "0.05"))

# {(provider, model) -> (embedder, {category: centroid})}. Module-level: the centroids
# depend only on CATEGORY_PROTOTYPES, so a process embeds them once and every later
# query in every later run reuses them. Keyed by the encoder because a different one
# puts the prototypes somewhere else entirely.
_CENTROID_CACHE = {}

# {(provider, model) -> monotonic deadline} for encoders that failed to build. Without
# it a misconfigured encoder is far WORSE than no encoder: building the centroids is 25
# prototype embeddings, and an unremembered failure re-attempts all 25 on every query,
# forever, in front of the LLM call it was supposed to replace. Measured against an
# out-of-credit OpenAI endpoint, every query paid a fresh failing round trip.
#
# A deadline rather than a permanent tombstone: the encoder is a separate service, and a
# blip during one query must not disable routing until the process restarts.
_ROUTER_RETRY_AFTER = {}
_ROUTER_COOLDOWN_S = float(os.getenv("SMART_RETRIEVER_ROUTER_COOLDOWN_S", "300"))


def _mean(vectors):
    """Centroid of equal-length vectors."""
    n = len(vectors)
    return [sum(col) / n for col in zip(*vectors)]


def _cos(a, b):
    """Cosine similarity. Local rather than imported from skills/tree_research: a
    retriever must not depend on a skill, and this is four lines."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _router(cfg):
    """``(embedder, {category: centroid})``, built once per process per encoder.

    The embedder is cached alongside the centroids, not rebuilt per query: constructing
    the langchain client re-reads config and re-opens the HTTP session, and measured over
    the 40-query set that cost 2.8s per query against 0.3s for the encode itself. The
    centroids depend only on CATEGORY_PROTOTYPES, so they are computed on the first
    query of the first run and reused by every query after it.

    Returns ``(None, None)`` when no embedding provider is configured -- which is what
    the unit configs present, so the classifier keeps its pre-router behaviour under test
    without the tests having to know this path exists.
    """
    provider = getattr(cfg, "embedding_provider", None)
    model = getattr(cfg, "embedding_model", None)
    if not provider or not model:
        return None, None

    key = (provider, model)
    if key in _CENTROID_CACHE:
        return _CENTROID_CACHE[key]
    if time.monotonic() < _ROUTER_RETRY_AFTER.get(key, 0.0):
        return None, None                    # failed recently; do not re-pay the build

    try:
        from ...memory.embeddings import Memory
        embedder = Memory(provider, model,
                          **getattr(cfg, "embedding_kwargs", {})).get_embeddings()
        # One batch call for all 25 prototypes rather than 25 round trips.
        flat = [q for cat in CATEGORY_PROTOTYPES for q in CATEGORY_PROTOTYPES[cat]]
        vectors = embedder.embed_documents(flat)
    except Exception as exc:
        _ROUTER_RETRY_AFTER[key] = time.monotonic() + _ROUTER_COOLDOWN_S
        logger.warning(f"embedding router unavailable ({exc}); routing by FAST_LLM for "
                       f"the next {_ROUTER_COOLDOWN_S:.0f}s")
        return None, None

    centroids, at = {}, 0
    for cat, prompts in CATEGORY_PROTOTYPES.items():
        centroids[cat] = _mean(vectors[at:at + len(prompts)])
        at += len(prompts)
    _CENTROID_CACHE[key] = (embedder, centroids)
    return embedder, centroids


def _embedding_category(cfg, query):
    """(category, margin) from the encoder, or None when it cannot or should not decide.

    None means "ask the LLM": no embedding provider, the service is down, or the top two
    categories are within `_EMBED_MARGIN` of each other. Every failure lands here, so a
    dead embedding server degrades to exactly the behaviour that shipped before this
    router existed -- never to a wrong bundle.

    Spawns no `claude` CLI session, so it is deliberately NOT wrapped in
    `agent_purpose("classify")`: that site reading 0 for a routed run is the signal that
    this router did the work.
    """
    embedder, centroids = _router(cfg)
    if not centroids:
        return None

    vector = embedder.embed_query(query)
    if not vector:
        return None

    ranked = sorted(((_cos(vector, c), cat) for cat, c in centroids.items()),
                    reverse=True)
    (best, category), (second, _) = ranked[0], ranked[1]
    margin = best - second
    if margin < _EMBED_MARGIN:
        return None
    return category, margin


# Map retriever names to the API key env vars they require.
# Retrievers not listed here (e.g. duckduckgo, arxiv, semantic_scholar) need no key.
_RETRIEVER_API_KEYS = {
    "tavily": "TAVILY_API_KEY",
    "exa": "EXA_API_KEY",
    "serper": "SERPER_API_KEY",
    "bing": "BING_API_KEY",
    "reddit": "FIRECRAWL_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY",
    "firecrawl_research": "FIRECRAWL_API_KEY",
}

# Retrievers that failed twice in a row in THIS process: a credential that answers
# 432 answers 432 for the next query too, so the route drops it instead of paying
# for the same failure once per sub-query. This is the second half of defect 1's
# "재시도/대체 라우팅": retry the call, and if the retriever still cannot serve,
# route around it permanently rather than warning about it forever.
#
# Every per-retriever failure — first, second, timeout — is a WARNING. The single
# ERROR is emitted by search(), and only when the query ends up with no results at
# all: spec/search-quality.md s1 (2026-07-27) defines retriever_errors as
# UNRECOVERED failures only, because the live probe counts ERROR records and
# counting failures a sibling already covered for would make retriever_errors == 0
# a verdict on the external services' mood rather than on this code.
#
# ponytail: module-level because SmartRetriever is constructed fresh per search
# call (actions/query_processing.get_search_results), so instance state could never
# remember. Per-process is the right lifetime — a new container run re-probes.
_DEAD_RETRIEVERS: set[str] = set()

# How long the parallel bundle waits before writing off whoever hasn't answered.
# Env-overridable so a slow deployment can widen it without a code change.
_RETRIEVER_TIMEOUT_S: float = float(os.getenv("SMART_RETRIEVER_TIMEOUT_S", "60"))

# Retrievers that have overrun the bundle timeout once. A single timeout is a
# blip (the same forgiveness _run_single_retriever's retry gives a raised error);
# the second one retires the retriever, because otherwise every remaining
# sub-query in the run re-pays the full timeout for the same dead service.
_TIMED_OUT_ONCE: set[str] = set()

# How many live retrievers a routed bundle must have before it is allowed to stand on
# its own. Below this it is topped up from general_web — see _route_to_retrievers. Three
# is what the smallest hand-written bundle in ROUTING_TABLE has, so this asks a degraded
# route for no more breadth than a healthy one already provides.
_MIN_ROUTED_RETRIEVERS = int(os.getenv("SMART_RETRIEVER_MIN_BUNDLE", "3"))


class SmartRetriever:
    """LLM-routed multi-retriever that selects optimal search engines per query."""

    def __init__(self, query, query_domains=None, researcher=None, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.researcher = researcher
        self.cfg = researcher.cfg if researcher else kwargs.get("cfg")
        # Retrievers that could not serve this query. Only read at the end of
        # search(), where it is finally known whether anything covered for them.
        self._failed_retrievers = []

    def search(self, max_results=10):
        """Classify the query, route to retrievers, execute in parallel, deduplicate."""
        try:
            category = self._classify_query()
            logger.info(f"SmartRetriever classified query as: {category}")

            retriever_configs = self._route_to_retrievers(category)
            if not retriever_configs and category != "general_web":
                logger.warning(f"No available retrievers for '{category}', falling back to general_web")
                retriever_configs = self._route_to_retrievers("general_web")
            if not retriever_configs:
                logger.warning("No available retrievers at all, trying tavily/duckduckgo as last resort")
                if self._check_retriever_availability("tavily"):
                    retriever_configs = [("tavily", max_results, {})]
                else:
                    retriever_configs = [("duckduckgo", max_results, {})]

            results = self._execute_retrievers(retriever_configs)
            results = self._deduplicate_results(results)
            if not results:
                tried = {entry[0] for entry in retriever_configs}
                results = self._deduplicate_results(
                    self._fallback_search(tried, max_results)
                )
            if not results:
                # The one ERROR this class emits: the query got nothing from anyone.
                # A failure a sibling or the fallback covered for stayed a WARNING.
                failed = sorted(set(self._failed_retrievers))
                logger.error(
                    f"SQ_RETRIEVAL_UNRECOVERED No results for query: {self.query} — "
                    f"unrecovered retriever failures: "
                    f"{failed or 'none (every retriever returned empty)'}"
                )
            return results[:max_results]
        except Exception as e:
            # Same marker: this query also ends with nothing usable, and an
            # uncounted crash is exactly the hollow zero the live gate exists to
            # catch. Mutually exclusive with the branch above, so still one record.
            logger.error(f"SQ_RETRIEVAL_UNRECOVERED SmartRetriever failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _classify_query(self):
        """Use FAST_LLM to classify the query into a routing category."""
        # Allow forced category override via config
        if self.cfg:
            force = getattr(self.cfg, "smart_retriever_force_category", None)
            if force and force in ROUTING_TABLE:
                return force

        if not self.cfg:
            return "general_web"

        # The encoder first. It costs no CLI session and no token, and it declines
        # (returns None) whenever it is not confident or not available -- so the LLM
        # below still runs for exactly the queries that need reasoning.
        try:
            routed = _embedding_category(self.cfg, self.query)
            if routed:
                category, margin = routed
                logger.info(f"SmartRetriever routed '{category}' by embedding "
                            f"(margin {margin:.3f}, no LLM call)")
                return category
        except Exception as e:
            logger.warning(f"Embedding router unavailable ({e}); asking the classifier LLM")

        try:
            from gpt_researcher.utils.llm import create_chat_completion
            from gpt_researcher.utils.agent_purpose import agent_purpose

            prompt = CLASSIFICATION_PROMPT.format(query=self.query)
            messages = [{"role": "user", "content": prompt}]

            # Run the async LLM call to completion. The MCP server invokes the
            # retriever from inside a running event loop, where the old
            # `loop.run_until_complete` raised "This event loop is already
            # running" — making classification ALWAYS fail and silently fall
            # back to general_web. `_run_coro_blocking` works in both contexts.
            # Tag OUTSIDE _run_coro_blocking, which copies this context into the worker
            # thread it spawns. This is the site the slimming targets: today it fires
            # once per sub-query per node, which is 4-5 of a node's 8-9 CLI sessions.
            with agent_purpose("classify"):
                response = _run_coro_blocking(
                    create_chat_completion(
                        model=self.cfg.fast_llm_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=20,
                        llm_provider=self.cfg.fast_llm_provider,
                        llm_kwargs=self.cfg.llm_kwargs,
                    )
                )

            category = str(response).strip().lower().replace(" ", "_")
            if category in ROUTING_TABLE:
                return category

            logger.warning(f"LLM returned unknown category '{category}', falling back to general_web")
            return "general_web"
        except Exception as e:
            logger.warning(f"Query classification failed: {e}. Falling back to general_web")
            return "general_web"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_to_retrievers(self, category):
        """Map category to available retriever configs, skipping unavailable ones."""
        configs = ROUTING_TABLE.get(category, ROUTING_TABLE["general_web"])

        # Check custom routing from config
        if self.cfg:
            custom = getattr(self.cfg, "smart_retriever_config", None)
            if custom and isinstance(custom, dict) and category in custom:
                configs = custom[category]

        available = []
        for entry in configs:
            name = entry[0]
            if self._check_retriever_availability(name):
                available.append(entry)
            else:
                logger.info(f"Skipping retriever '{name}': API key not configured")

        # A routed bundle that lost most of its members is worse than no routing at all.
        # Measured 2026-09-05: with smart routing restored, `code_technical` came down to
        # serper alone — exa retired itself ("Unable to import exa-py") and github
        # contributed nothing — so a technical query read 4 sources where the
        # general_web bundle had given it 10, and the report's citations fell from 11
        # to 4. Routing is supposed to make the search sharper, not narrower.
        #
        # So top the bundle up from general_web rather than replacing it: the routed
        # retrievers keep their priority and their specialism, and the breadth floor is
        # met by whatever else can actually serve (duckduckgo needs no key, so there is
        # always something). The existing fallback below only fires when the route
        # returns ZERO results, which is far too late to help here.
        if category != "general_web" and len(available) < _MIN_ROUTED_RETRIEVERS:
            routed = {entry[0] for entry in available}
            for entry in ROUTING_TABLE["general_web"]:
                if len(available) >= _MIN_ROUTED_RETRIEVERS:
                    break
                if entry[0] in routed or not self._check_retriever_availability(entry[0]):
                    continue
                logger.info("Topping up the '%s' bundle with '%s': only %d of its own "
                            "retrievers can serve", category, entry[0], len(routed))
                available.append(entry)

        return available

    # Tried in order when every routed retriever comes back empty (observed
    # outage: tavily 432 across the whole route -> silent total loss).
    # Keyless duckduckgo first, then keyed alternates — availability-checked,
    # so only retrievers whose API key is actually configured get tried.
    _FALLBACK_ORDER = ("duckduckgo", "tavily", "serper", "exa", "bing")

    def _fallback_search(self, tried, max_results):
        """Route to a not-yet-tried retriever instead of returning nothing."""
        for name in self._FALLBACK_ORDER:
            if name in tried or not self._check_retriever_availability(name):
                continue
            logger.warning(
                f"All routed retrievers returned 0 results; falling back to '{name}'"
            )
            results = self._run_single_retriever(name, max_results, {})
            if results:
                return results
        # No ERROR here: search() owns the single "this query got nothing" record.
        return []

    def _check_retriever_availability(self, name):
        """Return True if the retriever can serve: key configured and not proven dead."""
        if name in _DEAD_RETRIEVERS:
            return False
        env_var = _RETRIEVER_API_KEYS.get(name)
        if env_var is None:
            return True  # No key required
        return bool(os.environ.get(env_var))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_retrievers(self, configs):
        """Run retrievers in parallel using ThreadPoolExecutor."""
        all_results = []

        # NOT a `with` block: leaving one calls shutdown(wait=True), which joins the
        # very straggler the timeout below just wrote off. Measured cost of that join
        # on a live run — timeout logged at 06:26:19, call returned 06:31:29: 5m10s of
        # a 37-minute run spent waiting for results already declared lost.
        executor = ThreadPoolExecutor(max_workers=len(configs))
        try:
            futures = {}
            for name, max_res, extra_kwargs in configs:
                future = executor.submit(self._run_single_retriever, name, max_res, extra_kwargs)
                futures[future] = name

            try:
                for future in as_completed(futures, timeout=_RETRIEVER_TIMEOUT_S):
                    name = futures[future]
                    try:
                        results = future.result(timeout=30)
                        if results:
                            logger.info(f"Retriever '{name}' returned {len(results)} results")
                            all_results.extend(results)
                    except Exception as e:
                        # Not a retriever error: _run_single_retriever catches those
                        # itself and always returns a list. Reachable only when this
                        # individual future overruns result(timeout=30).
                        self._failed_retrievers.append(name)
                        logger.warning(f"Retriever '{name}' did not return in time: {e}")
            except TimeoutError:
                # Some futures didn't complete in time — keep partial results
                timed_out = [futures[f] for f in futures if not f.done()]
                self._failed_retrievers.extend(timed_out)
                self._retire_repeat_timeouts(timed_out)
                logger.warning(f"Retrievers timed out: {timed_out}. Returning partial results.")
        finally:
            # ponytail: cancel_futures drops queued work; a thread already inside a
            # blocking HTTP call cannot be cancelled and runs to completion in the
            # background. That is the cost of not waiting — bounded, because the
            # retriever's own retries terminate and a repeat offender gets retired.
            executor.shutdown(wait=False, cancel_futures=True)

        return all_results

    def _retire_repeat_timeouts(self, timed_out):
        """First timeout is forgiven, second retires the retriever from routing."""
        for name in timed_out:
            if name in _TIMED_OUT_ONCE:
                _DEAD_RETRIEVERS.add(name)
                logger.warning(
                    f"Retriever '{name}' timed out twice — retired from routing for "
                    f"this process; routing to an alternate"
                )
            else:
                _TIMED_OUT_ONCE.add(name)

    def _run_single_retriever(self, name, max_results, extra_kwargs):
        """Run a retriever, retrying once; a second failure retires it from routing.

        A non-retryable failure (400/401/403/404/422 — see retrievers.utils.
        is_retryable_error) skips that one free retry: the request itself is
        malformed/unauthorized/absent, so a retry would resend the identical
        request into the identical rejection — wall-clock spent on a guaranteed
        failure. Observed live: GithubSearch logged the same 422 six times, once
        per sub-query, because nothing here retired it after the first.
        """
        try:
            return self._invoke_retriever(name, max_results, extra_kwargs)
        except Exception as first:
            if not is_retryable_error(first):
                _DEAD_RETRIEVERS.add(name)
                self._failed_retrievers.append(name)
                logger.warning(
                    f"Retriever '{name}' failed with a non-retryable error ({first}) "
                    f"— retired from routing for this process without a wasted retry"
                )
                return []
            # The retry is defect 1's "재시도" half: a 432/timeout that a second call
            # clears never needed alternate routing, and reporting it as a failure
            # would turn every transient blip into a live-gate failure.
            logger.warning(f"Retriever '{name}' failed ({first}); retrying once")
        try:
            return self._invoke_retriever(name, max_results, extra_kwargs)
        except Exception as e:
            # WARNING, exactly once per retriever: whether this failure ends up
            # mattering is only knowable one frame up, after the whole parallel
            # bundle (and the fallback) resolves, so search() decides. Retiring the
            # retriever is the "대체 라우팅" half: the next sub-query routes to
            # something that works instead of re-paying for the same failure.
            _DEAD_RETRIEVERS.add(name)
            self._failed_retrievers.append(name)
            logger.warning(f"Retriever '{name}' failed twice ({e}) — retired from "
                           f"routing for this process; routing to an alternate")
            return []

    def _invoke_retriever(self, name, max_results, extra_kwargs):
        """Instantiate and run a single retriever. Raises on failure."""
        from gpt_researcher.actions.retriever import get_retriever

        retriever_class = get_retriever(name)
        if retriever_class is None:
            logger.warning(f"Unknown retriever: {name}")
            return []

        # copy: extra_kwargs is the dict living in ROUTING_TABLE, so popping from it
        # would strip the route's query_domains for every later query in the process
        extra_kwargs = dict(extra_kwargs)

        # Build constructor kwargs based on what the retriever accepts
        init_kwargs = {"query": self.query}

        # Pass query_domains: use route-specific override or the original
        route_domains = extra_kwargs.pop("query_domains", None)
        init_kwargs["query_domains"] = route_domains or self.query_domains

        # Pass remaining extra kwargs (topic, time_range, search_type, etc.)
        init_kwargs.update(extra_kwargs)

        # Filter kwargs to only those the constructor accepts
        import inspect
        sig = inspect.signature(retriever_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in init_kwargs.items() if k in valid_params}

        instance = retriever_class(**filtered)

        # Call search with max_results (some retrievers accept search_type in search())
        search_kwargs = {}
        search_sig = inspect.signature(instance.search)
        if "search_type" in search_sig.parameters and "search_type" in extra_kwargs:
            search_kwargs["search_type"] = extra_kwargs["search_type"]

        return instance.search(max_results=max_results, **search_kwargs) or []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_results(self, results):
        """Remove duplicate results by normalized URL, keeping the entry with longer body."""
        seen = {}
        for r in results:
            href = r.get("href", "")
            key = self._normalize_url(href)
            if key in seen:
                existing_body = seen[key].get("body", "")
                new_body = r.get("body", "")
                if len(new_body) > len(existing_body):
                    seen[key] = r
            else:
                seen[key] = r
        return list(seen.values())

    @staticmethod
    def _normalize_url(url):
        """Normalize URL for deduplication (strip scheme, trailing slash, fragment)."""
        try:
            parsed = urlparse(url)
            normalized = parsed.netloc + parsed.path.rstrip("/") + parsed.query
            return normalized.lower()
        except Exception:
            return url.lower()


def _run_coro_blocking(coro):
    """Run an async coroutine to completion from a sync function, whether or not
    the current thread already has a running event loop.

    SmartRetriever.search() is synchronous but is called from the MCP server's
    running event loop. Calling ``loop.run_until_complete`` there raises
    "This event loop is already running". When a loop is already running we run
    the coroutine on a fresh loop in a separate thread; otherwise we use
    ``asyncio.run`` directly.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — safe to drive one directly.
        return asyncio.run(coro)

    # A loop is already running in this thread: execute in a worker thread that
    # owns its own event loop so we don't touch the running one.
    from concurrent.futures import ThreadPoolExecutor
    import contextvars

    # Carry the caller's context across by hand. `ThreadPoolExecutor.submit` does NOT
    # copy contextvars, so the worker would otherwise start with an EMPTY context and
    # silently drop anything the caller set on one -- including the agent_purpose tag
    # that says which call site is spending the run's CLI-session budget. Measured
    # 2026-09-05: without this the var reads None in the worker, so every production
    # `classify` charge lands under "untagged" and the "classify + choose_agent <= 2
    # per run" gate passes vacuously by reading zero.
    #
    # This is the choke point all three branch-B callers route through (the MCP server
    # path plus the direct retriever_instance.search() calls in ResearchConductor), so
    # one copy here covers them. A future caller that spawns its own thread WITHOUT
    # coming through here would reintroduce the loss.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: ctx.run(asyncio.run, coro)).result()
