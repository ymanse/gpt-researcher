"""s14: one slow retriever must not take the whole MCP server down with it.

Measured 2026-09-15. The host's llama-server wedged into accepting TCP and never
answering, so `SmartRetriever`'s embedding route blocked on a socket read. Because
`get_search_results` is `async` but called the *synchronous* retriever inline, that read
happened ON the event loop thread: `/health` stopped answering (container went unhealthy,
FailingStreak 15) and two unrelated in-flight `deep_research` calls died with "MCP server
transport dropped mid-call" at 20m30s and 22m30s -- inside the openai client's 600s x 2
retries = 30 minute ceiling.

py-spy confirmed it, MainThread:
    read (httpcore/_backends/sync.py) <- embed_query <- _embedding_category
    <- _classify_query <- search <- get_search_results <- quick_search
    <- ... <- run_forever (asyncio/base_events.py)

So the retriever runs off the loop. These tests pin the two properties that fix needs:
the loop stays live while a retriever blocks, and the `agent_purpose` tag still rides
across the hop (a raw `ThreadPoolExecutor.submit` would silently drop it -- s11).
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gpt_researcher.actions.query_processing import get_search_results
from gpt_researcher.utils.agent_purpose import agent_purpose, current_purpose

BLOCK_S = 0.5          # long enough that a frozen loop cannot hide inside scheduler jitter


class BlockingRetriever:
    """A retriever that blocks its thread the way a hung HTTP read does."""

    ran_on: str | None = None
    saw_purpose: str | None = None

    def __init__(self, query, query_domains=None, researcher=None, **kwargs):
        self.query = query

    def search(self, max_results=10):
        BlockingRetriever.ran_on = threading.current_thread().name
        BlockingRetriever.saw_purpose = current_purpose()
        time.sleep(BLOCK_S)          # sync sleep: releases the GIL, never the loop
        return [{"href": "https://example.test/x", "body": "x"}]


@pytest.fixture(autouse=True)
def _reset():
    BlockingRetriever.ran_on = None
    BlockingRetriever.saw_purpose = None
    yield


async def test_the_loop_keeps_running_while_a_retriever_blocks():
    ticks = 0

    async def heartbeat():
        """Stands in for /health and the transport: it only advances if the loop is free."""
        nonlocal ticks
        while True:
            await asyncio.sleep(BLOCK_S / 10)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        results = await get_search_results("q", BlockingRetriever)
    finally:
        beat.cancel()

    assert results, "the retriever's results were dropped by the thread hop"
    assert ticks >= 3, (
        f"the event loop was serviced {ticks} times while a retriever blocked for "
        f"{BLOCK_S}s -- it should have ticked ~10 times. A blocked loop is what killed "
        f"/health and dropped the MCP transport on 2026-09-15")


async def test_the_retriever_does_not_run_on_the_loop_thread():
    loop_thread = threading.current_thread().name
    await get_search_results("q", BlockingRetriever)
    assert BlockingRetriever.ran_on is not None, "the retriever never ran"
    assert BlockingRetriever.ran_on != loop_thread, (
        f"the synchronous retriever ran on the loop thread ({loop_thread})")


async def test_the_call_site_tag_survives_the_hop():
    """s11's lesson: a thread hop that does not copy the context charges every LLM call
    to `untagged`, and the per-site budget gate then passes vacuously by reading zero."""
    with agent_purpose("classify"):
        await get_search_results("q", BlockingRetriever)
    assert BlockingRetriever.saw_purpose == "classify", (
        f"agent_purpose was lost across the thread hop: "
        f"{BlockingRetriever.saw_purpose!r}. asyncio.to_thread copies contextvars; a bare "
        f"ThreadPoolExecutor.submit does not")


async def test_two_searches_do_not_serialise_behind_each_other():
    started = time.monotonic()
    await asyncio.gather(*(get_search_results("q", BlockingRetriever) for _ in range(3)))
    elapsed = time.monotonic() - started
    assert elapsed < BLOCK_S * 2, (
        f"three concurrent searches took {elapsed:.2f}s for a {BLOCK_S}s block each -- "
        f"they ran one after another, which is what an inline sync call forces")
