"""s23: the run cannot oversubscribe its own embedding server.

Measured 2026-09-21 on the MCP container, the CoRet/SpIDER query at depth 2 with a
1500s budget:

  stage 1  7.3 min, 44-54KB of context per sub-query      -- healthy
  stage 2  17 min, NOTHING                                -- 4 of 6 nested researchers
           came back with a 2-character context, 2 were cancelled at the deadline
  log      48 APITimeoutError against 33 successful embedding calls

Nothing was misconfigured. DEEP_RESEARCH_CONCURRENCY bounds nested RESEARCHERS (4), and
each compresses its own 3-4 sub-queries concurrently, so 12-16 compressions hit one
llama.cpp server with 4 slots. Each waited out EMBEDDING_KWARGS' 300s request_timeout
and failed -- and ContextCompressor has no fallback, so a timed-out compression returns
"" and that sub-query's evidence is gone, silently, under a successful-looking run.

A longer timeout does not help: the requests queue behind each other either way. The
queueing has to happen on the CLIENT, where waiting is free, instead of on the server,
where waiting past the timeout destroys the evidence.

So this file pins:
  - concurrent embedding calls never exceed the configured limit;
  - the excess WAITS rather than failing -- every call still returns its vectors;
  - the limit is process-wide, so callers on other threads/loops share it;
  - 0 disables it (a hosted API that scales horizontally);
  - the wrapper is transparent: EmbeddingsFilter and friends see the same object.

Deterministic: no network, no real embedding server. "Concurrent" is measured by
overlap inside a fake encoder, not by wall time.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from langchain_core.embeddings import Embeddings

from gpt_researcher.memory import throttle
from gpt_researcher.memory.throttle import ThrottledEmbeddings, throttled


class _Encoder(Embeddings):
    """A fake embeddings object that records how many calls overlap inside it.

    `server_slots` is what the real one has: exceed it and the call fails the way an
    oversubscribed llama.cpp server does, after its request_timeout.
    """

    def __init__(self, server_slots: int | None = None, hold: float = 0.02):
        self.model = "fake-embedding-model"
        self.live = 0
        self.max_live = 0
        self.calls = 0
        self.failures = 0
        self._server_slots = server_slots
        self._hold = hold
        self._lock = threading.Lock()

    def _enter(self):
        with self._lock:
            self.live += 1
            self.calls += 1
            self.max_live = max(self.max_live, self.live)
            over = self._server_slots is not None and self.live > self._server_slots
            if over:
                self.failures += 1
        return over

    def _exit(self):
        with self._lock:
            self.live -= 1

    def embed_documents(self, texts, *args, **kwargs):
        over = self._enter()
        try:
            time.sleep(self._hold)
            if over:
                raise TimeoutError("Request timed out.")
            return [[0.1, 0.2] for _ in texts]
        finally:
            self._exit()

    def embed_query(self, text, *args, **kwargs):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts, *args, **kwargs):
        over = self._enter()
        try:
            await asyncio.sleep(self._hold)
            if over:
                raise TimeoutError("Request timed out.")
            return [[0.1, 0.2] for _ in texts]
        finally:
            self._exit()

    async def aembed_query(self, text, *args, **kwargs):
        return (await self.aembed_documents([text]))[0]


@pytest.fixture(autouse=True)
def fresh_semaphore(monkeypatch):
    """The semaphore is process-wide and cached; each test gets its own."""
    monkeypatch.setattr(throttle, "_semaphore", None)
    monkeypatch.setattr(throttle, "_semaphore_limit", None)
    yield


def _run_threads(fn, n, timeout: float = 30.0):
    """Run `fn` on `n` threads and fail -- rather than hang -- if they do not finish.

    Daemon threads and a bounded join on purpose. The bug class this file is about
    deadlocks: a blocking acquire on an event loop parks the very tasks holding the
    slots it waits for, so a plain `join()` never returns and pytest is killed by its
    runner instead of reporting a failure. (Measured while mutation-testing this file.)
    """
    threads = [threading.Thread(target=fn, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    stuck = [t for t in threads if t.is_alive()]
    assert not stuck, (
        f"{len(stuck)} of {n} embedding callers never finished within {timeout:.0f}s -- "
        "they are deadlocked waiting for slots, not queueing behind each other")


# ------------------------------------------------------------------ the defect itself

@pytest.mark.parametrize("limit,callers", [(4, 16), (2, 8)])
def test_the_excess_waits_instead_of_losing_its_evidence(monkeypatch, limit, callers):
    """The production failure, reproduced: more compressions than the server has slots.

    Without the limiter the encoder below fails every call above its slot count -- which
    is what `ContextCompressor` turns into an empty context and a silently missing
    sub-query. With it, every call must come back with vectors.
    """
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", str(limit))
    encoder = _Encoder(server_slots=limit)
    embeddings = throttled(encoder)
    results, errors = [], []

    def one():
        try:
            results.append(embeddings.embed_documents(["chunk"]))
        except Exception as e:  # noqa: BLE001 - the test records what the caller sees
            errors.append(e)

    _run_threads(one, callers)

    assert not errors, (
        f"{len(errors)} of {callers} embedding calls failed against a server with "
        f"{limit} slots ({errors[:1]}) -- each one is a sub-query whose evidence is "
        "silently dropped from the report")
    assert len(results) == callers and all(results), "a call returned no vectors"
    assert encoder.max_live <= limit, (
        f"{encoder.max_live} embedding calls were in flight at once against a limit of "
        f"{limit}: the limiter did not bound anything")
    assert encoder.calls == callers, (
        f"the encoder saw {encoder.calls} calls for {callers} callers -- the limiter "
        "dropped or duplicated work instead of queueing it")


def test_without_the_limiter_the_same_load_does_lose_evidence(monkeypatch):
    """Positive control. If the fake encoder tolerated oversubscription, the test above
    would pass with a no-op limiter."""
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "0")
    encoder = _Encoder(server_slots=4)
    embeddings = throttled(encoder)
    errors = []

    def one():
        try:
            embeddings.embed_documents(["chunk"])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    _run_threads(one, 16)

    assert errors, (
        "16 concurrent calls against 4 slots produced no failures with the limit "
        "disabled -- the fixture cannot show the defect this file is about")
    assert encoder.max_live > 4, "the unthrottled run never exceeded the slot count"


# ------------------------------------------------------------------ shape of the limit

def test_the_limit_is_shared_across_event_loops(monkeypatch):
    """SmartRetriever's router runs in worker threads with their own `asyncio.run`, so a
    loop-bound primitive would give each of them a private allowance -- which is how the
    run oversubscribed the server while every individual component looked bounded."""
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "2")
    encoder = _Encoder()
    embeddings = throttled(encoder)

    def own_loop():
        async def go():
            return await asyncio.gather(*(embeddings.aembed_query("q") for _ in range(3)))
        asyncio.run(go())

    _run_threads(own_loop, 4)

    assert encoder.max_live <= 2, (
        f"{encoder.max_live} calls overlapped across 4 independent event loops against a "
        "limit of 2 -- each loop got its own allowance")


def test_the_async_path_does_not_park_the_event_loop(monkeypatch):
    """The blocking acquire has to go to a thread. Parking the loop would stall every
    other coroutine in the run -- including the ones holding the slots it waits for,
    which is a deadlock, not a slowdown.

    Watched from ANOTHER thread on purpose. A parked loop cannot run its own timers, so
    `asyncio.wait_for` around the work never fires and the test hangs until the runner
    kills it -- a hang is not a reported failure. (Measured while mutation-testing this
    file: replacing the to_thread acquire with a blocking one killed the whole session.)
    """
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "1")
    embeddings = throttled(_Encoder(hold=0.05))
    done = threading.Event()
    state = {"ticks": 0}

    def work():
        async def heartbeat():
            while True:
                await asyncio.sleep(0.01)
                state["ticks"] += 1

        async def go():
            beat = asyncio.create_task(heartbeat())
            try:
                await asyncio.gather(*(embeddings.aembed_query("q") for _ in range(4)))
            finally:
                beat.cancel()

        try:
            asyncio.run(go())
        finally:
            done.set()

    # daemon: if the loop IS parked this thread never returns, and the suite must still
    # exit. Each test builds its own semaphore (see fresh_semaphore), so a thread left
    # holding a slot cannot reach the others.
    threading.Thread(target=work, daemon=True).start()

    assert done.wait(timeout=15), (
        "4 serialized embedding calls never finished: the event loop is parked inside a "
        "blocking acquire, waiting for a slot held by a task that loop is not running")
    assert state["ticks"] >= 3, (
        f"the event loop ticked {state['ticks']} times while the calls ran -- it was "
        "blocked rather than awaiting, so nothing else in the run could progress")


def test_zero_disables_the_limit(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "0")
    encoder = _Encoder()
    embeddings = throttled(encoder)
    _run_threads(lambda: embeddings.embed_documents(["chunk"]), 8)

    assert encoder.max_live > 1, (
        "calls were serialized although the limit is disabled -- a hosted embedding API "
        "that scales horizontally would be throttled to no purpose")


def test_a_garbage_limit_falls_back_instead_of_failing_the_run(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "not-a-number")
    embeddings = throttled(_Encoder())

    assert embeddings.embed_query("q"), (
        "a typo in EMBEDDING_MAX_CONCURRENCY killed the embedding call; a telemetry "
        "knob must not be able to fail a research run")


# ------------------------------------------------------------------ transparency

def test_the_wrapper_is_transparent_to_its_callers(monkeypatch):
    """`EmbeddingsFilter` and the retriever read attributes off this object and call the
    four methods; anything the wrapper hides would break a provider it never heard of."""
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "4")
    encoder = _Encoder()
    embeddings = throttled(encoder)

    assert embeddings.model == "fake-embedding-model", (
        "provider attributes are not delegated, so callers reading .model see nothing")
    assert embeddings.embed_query("q") == [0.1, 0.2]
    assert embeddings.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert asyncio.run(embeddings.aembed_query("q")) == [0.1, 0.2]
    assert throttled(embeddings) is embeddings, (
        "wrapping twice nests the limiter, so a call would take two slots and half the "
        "configured concurrency disappears")


def test_memory_hands_out_a_throttled_encoder(monkeypatch):
    """The wiring: every embedding in this process comes from Memory, which is why that
    is where the limit is applied."""
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "4")
    from gpt_researcher.memory.embeddings import Memory

    memory = Memory("openai", "text-embedding-3-small", openai_api_key="test-key")

    assert isinstance(memory.get_embeddings(), ThrottledEmbeddings), (
        "Memory returned a raw embeddings object, so every caller that goes through it "
        "-- the compressors, the router, the tree -- is unbounded again")


# ------------------------------------------------- the seam the unit tests did not reach

@pytest.mark.asyncio
async def test_a_throttled_encoder_still_drives_the_real_compressor(monkeypatch):
    """The integration this file originally missed, and it cost a live run.

    `EmbeddingsFilter` is a Pydantic model whose `embeddings` field is typed
    `Embeddings`, so a duck-typed wrapper is rejected at CONSTRUCTION -- "Input should
    be an instance of Embeddings". Measured 2026-09-21: every compression in a live run
    raised that ValidationError, each sub-query came back with an empty context, and the
    run still reported success with a report written from 225 characters.

    Every test above used the wrapper directly and passed throughout. This one puts it
    where production puts it.
    """
    monkeypatch.setenv("EMBEDDING_MAX_CONCURRENCY", "4")
    from gpt_researcher.context.compression import ContextCompressor

    documents = [
        {"url": f"https://example.invalid/{i}", "title": f"doc {i}",
         "raw_content": ("The outbox table grows without bound unless a cleanup job "
                         "trims it. " * 200)}
        for i in range(4)
    ]
    encoder = _Encoder()
    compressor = ContextCompressor(documents=documents, embeddings=throttled(encoder))

    context = await compressor.async_get_context("what bounds outbox growth?")

    assert context and context.strip(), (
        "the compressor produced no context with a throttled encoder -- in production "
        "this is a sub-query whose evidence vanishes while the run reports success")
    assert encoder.calls > 0, (
        "the compressor never reached the encoder, so this test would pass even if the "
        "embeddings object were ignored entirely")
