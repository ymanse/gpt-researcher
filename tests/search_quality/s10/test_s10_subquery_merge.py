"""s10 tests — every page a sub-query found reaches that sub-query, once.

Thresholds are the rig's measurements, not guesses. Baseline on the same fixture
(git checkout of researcher.py + compression.py, `python merge_bench.py`):

    metric                     before   after
    sub_query_source_coverage   0.333    1.0
    starved_sub_queries         1        0
    sources_in_context          1        5
    scrape_calls                5        5
    duplicate_blocks            0        0

`sources_in_context 1` before is the third defect the rig surfaced: the fast
compression path passed the raw page dict as Document.metadata, so
pretty_print_docs read metadata["source"] off a dict that spells it "url" and
every block printed "Source: None".

`duplicate_blocks 0` holds on both sides for different reasons — before, because
each page reached exactly one sub-query; after, because merge_sub_query_contexts
drops the repeat. Only the pair (coverage 1.0 AND duplicates 0) is the fix.
"""
import asyncio

import pytest

from gpt_researcher.context.compression import merge_sub_query_contexts

from merge_bench import run


@pytest.fixture(scope="module")
def metrics():
    # Sync on purpose: asyncio_mode is strict, where a module-scoped ASYNC fixture
    # needs pytest_asyncio.fixture AND a matching loop_scope on every consumer, and
    # the four consumers below never await anything — they only read this dict. One
    # asyncio.run here keeps the rig on the repo's convention (@pytest.mark.asyncio
    # marks genuinely async tests, nothing else).
    #
    # Not a latency assertion — sub-queries now wait on each other's page claims,
    # and a wait that never resolves must fail the suite instead of hanging it.
    return asyncio.run(asyncio.wait_for(run(), 30))


def test_every_sub_query_sees_every_page_it_found(metrics):
    assert metrics["sub_query_source_coverage"] == 1.0, (
        "a sub-query lost pages its own retrievers returned — visited_urls is "
        "acting as a read lock, not a fetch-once set (measured 0.333 before the "
        f"fix). per sub-query: {metrics['_per_sub_query']}"
    )


def test_no_sub_query_researches_nothing(metrics):
    assert metrics["starved_sub_queries"] == 0, (
        "a sub-query returned an empty context because siblings had claimed its "
        "whole result set; it then contributes nothing to the report"
    )


def test_coverage_is_not_bought_by_re_fetching(metrics):
    assert metrics["scrape_calls"] == metrics["sources_in_context"] == 5, (
        "each distinct page must be fetched exactly once and reach the context: "
        f"{metrics['scrape_calls']} fetches, {metrics['sources_in_context']} sources"
    )


def test_shared_pages_are_not_paid_for_twice(metrics):
    assert metrics["duplicate_blocks"] == 0, (
        "the same page reached several sub-queries and every copy survived into "
        "the merged context"
    )


# ---------------------------------------------------------------------------
# merge_sub_query_contexts on its own
# ---------------------------------------------------------------------------

def _ctx(*blocks: tuple[str, str]) -> str:
    return "\n".join(f"Source: {src}\nTitle: {src}\nContent: {body}\n"
                     for src, body in blocks)


def test_merge_drops_the_repeat_and_keeps_the_first():
    merged = merge_sub_query_contexts([
        _ctx(("u1", "shared fact"), ("u2", "only in a")),
        _ctx(("u1", "shared fact"), ("u3", "only in b")),
    ])

    assert merged.count("shared fact") == 1
    assert "only in a" in merged and "only in b" in merged


def test_merge_interleaves_so_truncation_never_starves_a_sub_query():
    """A budget that fits three blocks must spend it on three sub-queries, not on
    the first sub-query's first three."""
    merged = merge_sub_query_contexts(
        [_ctx(*((f"{name}{i}", f"{name} body {i}") for i in range(3)))
         for name in ("a", "b", "c")],
        max_chars=len(_ctx(("a0", "a body 0")).strip()) * 3 + 8,
    )

    assert [line.removeprefix("Source: ") for line in merged.splitlines()
            if line.startswith("Source: ")] == ["a0", "b0", "c0"]


def test_merge_of_nothing_is_nothing():
    assert merge_sub_query_contexts([]) == ""
    assert merge_sub_query_contexts(["", "   "]) == ""


def test_merge_passes_through_a_context_with_no_source_blocks():
    """Non-default prompt families do not emit "Source:" lines; the merge must
    degrade to concatenation rather than dropping the text."""
    merged = merge_sub_query_contexts(["granite formatted context", "another one"])

    assert "granite formatted context" in merged and "another one" in merged
