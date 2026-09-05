"""s11: routing must make the search sharper, not narrower.

Measured 2026-09-05, right after smart routing was restored on the main search path.
The run resolved `code_technical` and the bundle is exa + serper + github — but exa
retired itself in this image ("Unable to import exa-py") and github contributed nothing,
so the query went out to serper alone. It read 4 sources where the general_web bundle
had been giving it 10, and the report's citations fell from 11 to 4.

A route that has lost most of its members is worse than no route at all, and the
existing fallback cannot help: it fires only when the route returns ZERO results.

So a thin bundle is topped up from general_web — the routed retrievers keep their
priority and their specialism, and the breadth floor is met by whatever else can serve.
duckduckgo needs no key, so there is always something.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gpt_researcher.retrievers.smart.smart_retriever as sr_mod
from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE, SmartRetriever


@pytest.fixture(autouse=True)
def clean_process_state(monkeypatch):
    """_DEAD_RETRIEVERS is module-level and per-process; never leak it to a sibling."""
    monkeypatch.setattr(sr_mod, "_DEAD_RETRIEVERS", set())
    yield


def _retriever(**cfg_over):
    cfg = SimpleNamespace(smart_retriever_config=None,
                          smart_retriever_force_category=None, **cfg_over)
    return SmartRetriever("does the outbox pattern deduplicate replays?", cfg=cfg)


def _only_keyless(monkeypatch):
    """No API keys at all: duckduckgo and the keyless retrievers are all that survive."""
    for env in set(sr_mod._RETRIEVER_API_KEYS.values()):
        monkeypatch.delenv(env, raising=False)


def test_a_bundle_gutted_by_missing_keys_is_topped_up(monkeypatch):
    _only_keyless(monkeypatch)
    # code_technical is exa(key) + serper(key) + github(keyless): only github survives
    routed = _retriever()._route_to_retrievers("code_technical")
    names = [entry[0] for entry in routed]

    assert "github" in names, (
        f"the routed bundle's own surviving retriever was dropped: {names}")
    # The floor is best effort, not a guarantee: with no keys at all, duckduckgo is the
    # only general_web retriever that can serve, so two is everything there is. What must
    # hold is that the top-up took EVERY available one rather than leaving the route thin.
    reachable = [e[0] for e in ROUTING_TABLE["general_web"]
                 if e[0] not in ("github",)
                 and _retriever()._check_retriever_availability(e[0])]
    assert "duckduckgo" in names, (
        f"the keyless general_web retriever was not added to a gutted bundle: {names}. "
        f"That is the measured regression: a technical query read 4 sources where "
        f"general_web had given it 10, and citations fell from 11 to 4")
    assert len(names) == 1 + min(len(reachable), sr_mod._MIN_ROUTED_RETRIEVERS - 1), (
        f"code_technical came down to {names}; every available general_web retriever "
        f"({reachable}) should have been added up to the floor of "
        f"{sr_mod._MIN_ROUTED_RETRIEVERS}")
    assert names[:1] == ["github"], (
        f"the top-up must come AFTER the routed retrievers, not displace them: {names}")


def test_a_retired_retriever_counts_as_missing(monkeypatch):
    """exa did not lack a key — it failed to import and retired itself mid-run."""
    monkeypatch.setenv("EXA_API_KEY", "x")
    monkeypatch.setenv("SERPER_API_KEY", "x")
    monkeypatch.setattr(sr_mod, "_DEAD_RETRIEVERS", {"exa"})

    names = [e[0] for e in _retriever()._route_to_retrievers("code_technical")]
    assert "exa" not in names, f"a retired retriever was routed to again: {names}"
    assert len(names) >= sr_mod._MIN_ROUTED_RETRIEVERS, (
        f"a bundle thinned by a RETIRED retriever was left thin: {names}. Availability "
        f"already accounts for _DEAD_RETRIEVERS, so the floor must too")


def test_a_healthy_bundle_is_left_exactly_as_routed(monkeypatch):
    """No top-up when the route can serve on its own — routing must stay meaningful."""
    for env in set(sr_mod._RETRIEVER_API_KEYS.values()):
        monkeypatch.setenv(env, "x")

    names = [e[0] for e in _retriever()._route_to_retrievers("code_technical")]
    assert names == [e[0] for e in ROUTING_TABLE["code_technical"]], (
        f"a fully available bundle was modified: {names}")


def test_general_web_is_never_topped_up_with_itself(monkeypatch):
    _only_keyless(monkeypatch)
    names = [e[0] for e in _retriever()._route_to_retrievers("general_web")]
    assert len(names) == len(set(names)), f"duplicate retrievers routed: {names}"
