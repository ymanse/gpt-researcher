"""One process-wide limit on how many embedding requests are in flight at once.

Measured 2026-09-21 on the MCP container, a paper-heavy query at depth 2: the stage-1
round produced 44-54KB of context per sub-query, and the stage-2 round produced NOTHING
-- four of its six nested researchers came back with a 2-character context and the other
two were cancelled at the deadline. The log holds 48 `APITimeoutError` from the local
embedding server against 33 successful calls.

Nothing was misconfigured. The concurrency knob bounds nested RESEARCHERS (4), and each
of those compresses its own 3-4 sub-queries at the same time, so 12-16 compressions hit
one llama.cpp server with 4 slots. Each waits out EMBEDDING_KWARGS' 300s request_timeout
and then fails -- and `ContextCompressor` has no fallback, so a timed-out compression
returns "" and that sub-query's evidence is gone. Silently: the run reports success, the
report is written from whatever survived, and the reader cannot tell.

The fix is not a longer timeout -- the requests are queued behind each other either way.
It is to do the queueing HERE, where waiting is free, instead of on the server, where
waiting past 300s destroys the evidence. Total work is unchanged; a run gets slower
rather than emptier.

WHY A SEMAPHORE AND NOT AN EXECUTOR: the calls this bounds are synchronous
(`EmbeddingsFilter` -> `embed_documents`, driven through `asyncio.to_thread`), and they
come from three places -- the compressors, SmartRetriever's router (its own worker
threads, its own event loop) and the tree's `aembed_query`. A `threading.BoundedSemaphore`
is the one primitive all three can share: it is not bound to an event loop, so a thread
that runs `asyncio.run` of its own still contends on the same counter.

Blocking a worker thread here is deliberate. `asyncio.to_thread` hands work to the
default executor (min(32, cpu+4) = 24 threads on this container), so a few parked
threads cost nothing; the alternative -- letting the call through to time out -- costs
the sub-query.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# How many embedding requests may be in flight process-wide. The default matches the
# llama.cpp server's slot count (4, measured on the RTX 4060 host): more than that does
# not make the server faster, it only moves the queue to where a timeout can kill it.
# 0 disables the limit -- the pre-2026-09-21 behaviour, and what a hosted embedding API
# that scales horizontally would want.
DEFAULT_MAX_CONCURRENCY = 4

# How long a call may WAIT for a slot before giving up on the limiter and going through
# anyway. A deadlock here would be worse than the oversubscription this prevents: the
# call still has its own request_timeout, so letting it through degrades to today's
# behaviour rather than hanging the run.
ACQUIRE_TIMEOUT_S = 600.0


def _limit() -> int:
    try:
        return int(os.environ.get("EMBEDDING_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
    except ValueError:
        logger.warning("EMBEDDING_MAX_CONCURRENCY is not a number, using %d",
                       DEFAULT_MAX_CONCURRENCY)
        return DEFAULT_MAX_CONCURRENCY


_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_limit: int | None = None


def _get_semaphore() -> threading.BoundedSemaphore | None:
    """The process-wide semaphore, or None when the limit is disabled.

    Built on first use rather than at import: the limit is read from the environment,
    and `server.py` sets those variables per MCP tool call.
    """
    global _semaphore, _semaphore_limit
    limit = _limit()
    if limit <= 0:
        return None
    with _lock:
        if _semaphore is None or _semaphore_limit != limit:
            _semaphore = threading.BoundedSemaphore(limit)
            _semaphore_limit = limit
            logger.info("embedding concurrency limited to %d in-flight request(s)", limit)
        return _semaphore


class ThrottledEmbeddings(Embeddings):
    """An embeddings object that holds a slot for the duration of each call.

    Wraps the provider's object rather than subclassing it: LangChain embeddings are
    Pydantic models with provider-specific fields, and this has to work for every
    provider `Memory` builds. Everything not named here is delegated, so callers that
    read `.model` see no difference.

    It DOES subclass langchain_core's `Embeddings` interface, and that is load-bearing,
    not decoration: `EmbeddingsFilter` is a Pydantic model whose `embeddings` field is
    typed `Embeddings`, so a plain duck-typed wrapper is rejected at construction --
    "Input should be an instance of Embeddings". Measured 2026-09-21: with the wrapper
    not subclassing, every compression in a live run raised that ValidationError, each
    sub-query returned an empty context, and the run still reported success. The
    interface is an ABC (not a Pydantic model), so inheriting costs nothing.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"ThrottledEmbeddings({self._inner!r})"

    def _call(self, method: str, *args, **kwargs):
        semaphore = _get_semaphore()
        if semaphore is None:
            return getattr(self._inner, method)(*args, **kwargs)
        waited = time.monotonic()
        if not semaphore.acquire(timeout=ACQUIRE_TIMEOUT_S):
            # Never fail the call over the limiter: it exists to protect the request,
            # not to police it.
            logger.warning("waited %.0fs for an embedding slot and gave up; calling %s "
                           "unthrottled", ACQUIRE_TIMEOUT_S, method)
            return getattr(self._inner, method)(*args, **kwargs)
        delay = time.monotonic() - waited
        if delay > 5:
            logger.info("embedding call %s waited %.0fs for a slot", method, delay)
        try:
            return getattr(self._inner, method)(*args, **kwargs)
        finally:
            semaphore.release()

    async def _acall(self, method: str, *args, **kwargs):
        # The async path runs on an event loop, so the blocking acquire goes to a
        # thread -- parking the LOOP here would stall every other coroutine in the run,
        # including the ones holding slots.
        import asyncio

        semaphore = _get_semaphore()
        if semaphore is None:
            return await getattr(self._inner, method)(*args, **kwargs)
        acquired = await asyncio.to_thread(semaphore.acquire, True, ACQUIRE_TIMEOUT_S)
        if not acquired:
            logger.warning("waited %.0fs for an embedding slot and gave up; calling %s "
                           "unthrottled", ACQUIRE_TIMEOUT_S, method)
            return await getattr(self._inner, method)(*args, **kwargs)
        try:
            return await getattr(self._inner, method)(*args, **kwargs)
        finally:
            semaphore.release()

    # The four methods every call site in this codebase uses. Named explicitly rather
    # than generated, so a provider that lacks one raises AttributeError from the
    # delegate as it always did.
    def embed_documents(self, texts, *args, **kwargs):
        return self._call("embed_documents", texts, *args, **kwargs)

    def embed_query(self, text, *args, **kwargs):
        return self._call("embed_query", text, *args, **kwargs)

    async def aembed_documents(self, texts, *args, **kwargs):
        return await self._acall("aembed_documents", texts, *args, **kwargs)

    async def aembed_query(self, text, *args, **kwargs):
        return await self._acall("aembed_query", text, *args, **kwargs)


def throttled(embeddings: Any) -> Any:
    """`embeddings`, wrapped so its calls share the process-wide slot count."""
    if embeddings is None or isinstance(embeddings, ThrottledEmbeddings):
        return embeddings
    return ThrottledEmbeddings(embeddings)
