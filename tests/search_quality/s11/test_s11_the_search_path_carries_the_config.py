"""s11: the retriever on the web-search path must be given the researcher's config.

Found 2026-09-05 by reading the attribution P0 added, against a live run:

    tree run routing category resolved once: code_technical
    SmartRetriever classified query as: general_web

`ResearchConductor._search_relevant_source_urls` builds its retriever as
`retriever_class(query, query_domains=query_domains)` — with no `researcher=`. So
`SmartRetriever.__init__` stores `cfg = None`, and `_classify_query` takes its very
first branch (`if not self.cfg: return "general_web"`) and returns without asking
anything. Two consequences, one pre-existing and one new:

  * The fork's own smart routing has been inert on the main search path. Every query,
    academic or code or news, routes to the general_web bundle; arxiv, semantic_scholar,
    github and exa are never reached from here no matter what the question is.
  * P1.1 cannot work. The run resolves a category once and stamps it onto each node's
    cfg, but the retriever never sees that cfg, so the stamp is discarded.

The sibling s11 classify tests do not catch this: their fake node researcher calls
`SmartRetriever(sub_query, researcher=self)` itself, which is the call the real code
does NOT make.

Passing the researcher costs no CLI session when a category is forced — that is the
branch it enables — and restores the routing the retriever was written for.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.skills.researcher as researcher_mod
from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE
from gpt_researcher.skills.researcher import ResearchConductor

QUERY = "How do transactional outbox implementations handle duplicate delivery?"
FORCED = "code_technical"
assert FORCED in ROUTING_TABLE


class _RecordingRetriever:
    """Stands in for SmartRetriever, keeping only what this contract is about."""

    seen: list = []

    def __init__(self, query, query_domains=None, researcher=None, **kwargs):
        self.query = query
        # exactly how SmartRetriever resolves its config
        self.cfg = researcher.cfg if researcher else kwargs.get("cfg")
        _RecordingRetriever.seen.append(self)

    def search(self, max_results=10):
        return []


def _cfg():
    return SimpleNamespace(
        max_search_results_per_query=5,
        max_iterations=3,
        smart_retriever_force_category=FORCED,
        smart_retriever_config=None,
        fast_llm_provider="fake", fast_llm_model="fast",
        llm_kwargs={},
    )


def _conductor():
    _RecordingRetriever.seen = []
    researcher = SimpleNamespace(
        query=QUERY,
        cfg=_cfg(),
        retrievers=[_RecordingRetriever],
        visited_urls=set(),
        research_sources=[],
        add_research_sources=lambda s: None,
        verbose=False,
        websocket=None,
        query_domains=[],
        report_type="research_report",
        report_source="web",
        vector_store=None,
        headers={},
    )
    return ResearchConductor(researcher), researcher


@pytest.mark.asyncio
async def test_the_web_search_retriever_is_built_with_the_researcher():
    """The retriever must be able to read cfg — otherwise routing dies silently."""
    conductor, researcher = _conductor()

    with mock.patch.object(researcher_mod, "stream_output", new=mock.AsyncMock()):
        await conductor._search_relevant_source_urls(QUERY)

    assert _RecordingRetriever.seen, (
        "fixture precondition: the search path must build a retriever at all")
    without_cfg = [r for r in _RecordingRetriever.seen if r.cfg is None]
    assert not without_cfg, (
        f"{len(without_cfg)} of {len(_RecordingRetriever.seen)} retrievers on the web "
        "search path were built without the researcher, so SmartRetriever stores "
        "cfg=None and _classify_query returns 'general_web' on its first branch without "
        "asking anything. Smart routing is inert on this path and the run's resolved "
        "category never reaches the search"
    )


@pytest.mark.asyncio
async def test_the_forced_category_actually_reaches_the_search_path():
    """The end-to-end consequence: what the run resolved is what the search routes by."""
    conductor, researcher = _conductor()

    with mock.patch.object(researcher_mod, "stream_output", new=mock.AsyncMock()):
        await conductor._search_relevant_source_urls(QUERY)

    routed = [getattr(r.cfg, "smart_retriever_force_category", None)
              for r in _RecordingRetriever.seen]
    assert routed and all(c == FORCED for c in routed), (
        f"the retrievers on the search path see {routed} instead of {FORCED!r}. The tree "
        "resolves the routing category once and stamps it on every node researcher's "
        "cfg; if the retriever is not given that researcher the stamp is discarded and "
        "every node searches the general_web bundle regardless of the question"
    )
