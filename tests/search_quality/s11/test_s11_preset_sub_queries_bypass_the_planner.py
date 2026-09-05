"""s11 / P1.3: a preset sub-query list must bypass the planner entirely.

Measured 2026-09-05 (harness-search/spec/p0-p1-slimming.md): one tree node costs 8-9
`claude` CLI sessions and 5-6 of them buy nothing. Two of the wasted purchases belong to
the planner: `plan_research` runs a probe search and then spends a `plan` LLM call to
invent sub-queries for a question the tree ALREADY decomposed. The node question IS the
plan, so both are paid for a decomposition that is thrown away.

P1.3 therefore gives `GPTResearcher` a `preset_sub_queries` list, and
`ResearchConductor._get_context_by_web_search` researches it INSTEAD of calling
`plan_research`. Three properties are easy to get wrong and are pinned here:

  - it must be a NAMED constructor parameter. Anything reaching `**kwargs` lands in
    `self.kwargs`, and `plan_research` splats `**self.researcher.kwargs` straight into
    `plan_research_outline` — i.e. a kwargs-carried preset is pasted into an LLM call.
  - the existing code appends the original query whenever
    `report_type != "subtopic_report"`. With a preset the researcher's query IS the node
    question, so a naive `sub_queries = preset; sub_queries.append(query)` researches
    that string twice — buying back one of the searches the bypass was meant to save.
  - it must be read with `getattr(..., "preset_sub_queries", None)`. Callers that
    predate P1.3 hand the conductor a researcher with no such attribute at all —
    `tests/search_quality/s10/merge_bench.py` is one — and a bare attribute read
    kills them with an AttributeError. Every fake below sets the attribute
    explicitly, so only
    `test_a_researcher_that_never_defines_the_attribute_plans_instead_of_raising`
    can see that.

Hermetic: no network, no LLM, no embeddings service. The planner's probe search
(`get_search_results`) and the planner LLM call (`plan_research_outline`) are recorded at
their import site in `gpt_researcher.skills.researcher`, `Memory` is stubbed so no
embeddings client is built, and `_process_sub_query` is replaced by a recorder.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import gpt_researcher.agent as agent_mod
import gpt_researcher.skills.researcher as researcher_mod
from gpt_researcher.agent import GPTResearcher
from gpt_researcher.skills.researcher import ResearchConductor


QUERY = "How do transactional outbox implementations handle duplicate delivery?"
PRESET = [
    "Outbox consumer deduplication strategies in production systems",
    "Idempotency keys for at-least-once outbox delivery",
    QUERY,  # the tree hands the node its own question back - must not be researched twice
]
PLANNED = ["planner sub-query one", "planner sub-query two"]


class _PlainWebRetriever:
    """`_get_context_by_web_search` only reads `__name__` to spot MCP retrievers."""


def _fake_researcher(**overrides):
    """The attribute surface `_get_context_by_web_search` and `plan_research` read.

    Same shape the s4/s8 tests use for TreeResearchSkill's parent: a namespace, not a
    real agent, so the conductor runs without a Config or an embeddings client.
    """
    fields = dict(
        query=QUERY,
        report_type="research_report",
        retrievers=[_PlainWebRetriever],
        cfg=SimpleNamespace(),
        verbose=False,
        websocket=None,
        role="",
        parent_query="",
        query_domains=[],
        visited_urls=set(),
        kwargs={},
        add_costs=lambda *a, **k: None,
        preset_sub_queries=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def planner_probes(monkeypatch):
    """Record the planner's two purchases: the probe search and the planning LLM call."""
    probes: list[str] = []
    plans: list[dict] = []

    async def _probe(query, retriever, query_domains=None, researcher=None):
        probes.append(query)
        return []

    async def _plan(**kwargs):
        plans.append(kwargs)
        return list(PLANNED)

    monkeypatch.setattr(researcher_mod, "get_search_results", _probe)
    monkeypatch.setattr(researcher_mod, "plan_research_outline", _plan)
    return SimpleNamespace(probes=probes, plans=plans)


def _record_sub_queries(conductor, monkeypatch) -> list[str]:
    """Replace the per-sub-query research pass with a recorder.

    Arity matters: `_get_context_by_web_search` calls this inside an `asyncio.gather`
    wrapped in `except Exception: return []`, so a signature mismatch would be swallowed
    and would look like "the planner was skipped and nothing was researched".
    """
    researched: list[str] = []

    async def _process(sub_query, scraped_data=[], query_domains=[]):
        researched.append(sub_query)
        return f"Source: https://example.test/{len(researched)}\ncontext for {sub_query}"

    monkeypatch.setattr(conductor, "_process_sub_query", _process)
    return researched


def _real_researcher(monkeypatch, **kwargs) -> GPTResearcher:
    """A real GPTResearcher minus the embeddings client, which needs a live API key."""
    monkeypatch.setattr(
        agent_mod, "Memory",
        lambda *a, **k: SimpleNamespace(get_embeddings=lambda: None),
    )
    return GPTResearcher(query=QUERY, verbose=False, **kwargs)


def test_preset_sub_queries_is_a_named_constructor_parameter_not_a_kwarg(monkeypatch):
    """A kwargs-carried preset lands in `self.kwargs`, which `plan_research` splats into
    the planner call. A named parameter is the only shape that cannot leak."""
    params = inspect.signature(GPTResearcher.__init__).parameters
    assert "preset_sub_queries" in params, (
        "GPTResearcher.__init__ has no `preset_sub_queries` parameter, so any caller "
        f"passing one feeds **kwargs. Named parameters today: {sorted(params)}"
    )
    assert params["preset_sub_queries"].default is None, (
        "the default must be None (no preset, plan as today), got "
        f"{params['preset_sub_queries'].default!r}"
    )

    researcher = _real_researcher(monkeypatch, preset_sub_queries=list(PRESET))
    assert researcher.preset_sub_queries == PRESET, (
        f"the preset must be kept verbatim, got {researcher.preset_sub_queries!r}"
    )
    assert "preset_sub_queries" not in researcher.kwargs, (
        "the preset was absorbed into self.kwargs, which plan_research forwards into the "
        f"planner LLM call. self.kwargs={researcher.kwargs!r}"
    )


@pytest.mark.asyncio
async def test_the_preset_is_never_forwarded_into_the_planner_llm_call(
        monkeypatch, planner_probes):
    """The concrete damage a kwargs-carried preset does: `plan_research` calls
    `plan_research_outline(..., **self.researcher.kwargs)`, so the whole sub-query list
    is handed to the prompt builder as an unrecognised keyword."""
    researcher = _real_researcher(monkeypatch, preset_sub_queries=list(PRESET))

    await researcher.research_conductor.plan_research(QUERY)

    assert planner_probes.plans, "fixture check: the planner call must have been recorded"
    forwarded = planner_probes.plans[-1]
    assert "preset_sub_queries" not in forwarded, (
        "the preset reached the planner LLM call as a forwarded keyword - it is research "
        f"scaffolding, not prompt input. forwarded keys: {sorted(forwarded)}"
    )


@pytest.mark.asyncio
async def test_a_non_empty_preset_skips_both_the_planner_and_its_probe_search(
        monkeypatch, planner_probes):
    """The saving is two purchases per node, not one: `plan_research` runs a probe search
    BEFORE it spends the planning LLM call, and both must go."""
    conductor = ResearchConductor(_fake_researcher(preset_sub_queries=list(PRESET)))
    researched = _record_sub_queries(conductor, monkeypatch)

    await conductor._get_context_by_web_search(QUERY, [], [])

    assert researched, (
        "fixture check: the preset path must still research something - an empty result "
        "here would mean the gather raised and was swallowed, not that the planner was skipped"
    )
    assert planner_probes.probes == [], (
        "the planner's probe search still ran with a preset in hand; it costs a retriever "
        f"round-trip for a decomposition that already exists. probe queries: {planner_probes.probes}"
    )
    assert planner_probes.plans == [], (
        "the planning LLM call still ran with a preset in hand - that is the `plan` call "
        f"site P1.3 removes. planner calls: {len(planner_probes.plans)}"
    )


@pytest.mark.asyncio
async def test_the_preset_list_is_exactly_what_gets_researched(monkeypatch, planner_probes):
    """Skipping the planner only helps if the preset replaces it: every entry must reach
    `_process_sub_query`, and nothing invented may join them."""
    conductor = ResearchConductor(_fake_researcher(preset_sub_queries=list(PRESET)))
    researched = _record_sub_queries(conductor, monkeypatch)

    await conductor._get_context_by_web_search(QUERY, [], [])

    assert sorted(researched) == sorted(PRESET), (
        "the sub-queries actually researched are not the preset. expected the preset "
        f"{PRESET}, measured {researched}"
    )


@pytest.mark.asyncio
async def test_a_preset_entry_equal_to_the_researchers_own_query_is_researched_once(
        monkeypatch, planner_probes):
    """`_get_context_by_web_search` appends the original query for every non-subtopic
    report. With a preset the node question IS the researcher's query, so appending it
    unconditionally pays for the same search, scrape and compression pass twice."""
    conductor = ResearchConductor(_fake_researcher(preset_sub_queries=[QUERY]))
    researched = _record_sub_queries(conductor, monkeypatch)

    await conductor._get_context_by_web_search(QUERY, [], [])

    assert researched == [QUERY], (
        "a one-entry preset holding the researcher's own query must be researched exactly "
        f"once and nothing else added. measured {researched} - "
        f"{researched.count(QUERY)} pass(es) over that one string"
    )


@pytest.mark.asyncio
async def test_the_default_preset_is_none_and_still_plans_exactly_as_today(
        monkeypatch, planner_probes):
    """P1.3 must be opt-in: with no preset the linear researcher keeps buying the probe
    search, the planning call and the appended original query, byte for byte."""
    researcher = _real_researcher(monkeypatch)
    assert researcher.preset_sub_queries is None, (
        "the no-preset default must be None so the planner path is untouched, got "
        f"{researcher.preset_sub_queries!r}"
    )

    conductor = researcher.research_conductor
    researched = _record_sub_queries(conductor, monkeypatch)

    await conductor._get_context_by_web_search(QUERY, [], [])

    assert planner_probes.probes == [QUERY], (
        "without a preset the planner's probe search must still run once for the query, "
        f"measured {planner_probes.probes}"
    )
    assert len(planner_probes.plans) == 1, (
        "without a preset the planning LLM call must still run exactly once, measured "
        f"{len(planner_probes.plans)}"
    )
    assert researched == PLANNED + [QUERY], (
        "without a preset the researched list must stay planned-sub-queries + original "
        f"query. expected {PLANNED + [QUERY]}, measured {researched}"
    )


@pytest.mark.asyncio
async def test_an_empty_preset_falls_back_to_planning_while_a_non_empty_one_does_not(
        monkeypatch, planner_probes):
    """Emptiness, not presence, decides. `preset_sub_queries=[]` means "I have no
    decomposition for you", so it must plan - treating it as "research nothing" would
    return an empty context and fail the node on MIN_CONTEXT_CHARS."""
    empty_conductor = ResearchConductor(_fake_researcher(preset_sub_queries=[]))
    empty_researched = _record_sub_queries(empty_conductor, monkeypatch)
    await empty_conductor._get_context_by_web_search(QUERY, [], [])

    assert empty_researched == PLANNED + [QUERY], (
        "an empty preset must fall back to planning, not research nothing. expected "
        f"{PLANNED + [QUERY]}, measured {empty_researched}"
    )
    plans_after_empty = len(planner_probes.plans)

    filled_conductor = ResearchConductor(_fake_researcher(preset_sub_queries=list(PRESET)))
    filled_researched = _record_sub_queries(filled_conductor, monkeypatch)
    await filled_conductor._get_context_by_web_search(QUERY, [], [])

    assert len(planner_probes.plans) == plans_after_empty, (
        "an empty and a non-empty preset were treated the same way: measured "
        f"{plans_after_empty} planner call(s) for the empty preset and "
        f"{len(planner_probes.plans) - plans_after_empty} for the non-empty one, which "
        f"researched {filled_researched}"
    )


@pytest.mark.asyncio
async def test_a_researcher_that_never_defines_the_attribute_plans_instead_of_raising(
        monkeypatch, planner_probes):
    """The preset must be read with `getattr(..., "preset_sub_queries", None)`.

    Every other test in this file hands the conductor a fake that sets the attribute
    explicitly, so none of them can tell a bare `self.researcher.preset_sub_queries`
    read from a defaulted one. `tests/search_quality/s10/merge_bench.py` can: it drives
    this same method over a `SimpleNamespace` researcher that never defines the
    attribute, so a bare read takes the s10 bench down with an AttributeError - a
    green-looking s11 paid for by a broken s10.
    """
    # Precondition. A code path that reads the attribute nowhere cannot trip over its
    # absence, so the absence rule is unfalsifiable until the preset branch exists.
    preset_conductor = ResearchConductor(_fake_researcher(preset_sub_queries=list(PRESET)))
    _record_sub_queries(preset_conductor, monkeypatch)
    await preset_conductor._get_context_by_web_search(QUERY, [], [])
    assert planner_probes.plans == [] and planner_probes.probes == [], (
        "P1.3 is not implemented: the planner still ran for a researcher carrying a "
        "preset, so nothing reads `preset_sub_queries` yet and a bare attribute read is "
        f"indistinguishable from a getattr here. measured {len(planner_probes.probes)} "
        f"probe search(es) and {len(planner_probes.plans)} planner call(s)"
    )

    bare = _fake_researcher()
    del bare.preset_sub_queries
    assert not hasattr(bare, "preset_sub_queries"), (
        "fixture check: this researcher must genuinely lack the attribute, mirroring "
        "s10's SimpleNamespace - setting it to None would test nothing"
    )
    conductor = ResearchConductor(bare)
    researched = _record_sub_queries(conductor, monkeypatch)

    try:
        await conductor._get_context_by_web_search(QUERY, [], [])
    except AttributeError as exc:
        if "preset_sub_queries" not in str(exc):
            raise
        pytest.fail(
            "`preset_sub_queries` is read as a bare attribute, so a researcher that "
            "never defines it dies instead of planning. Any caller that predates P1.3 "
            "has this shape, s10/merge_bench.py included. raised: "
            f"{type(exc).__name__}: {exc}"
        )

    assert planner_probes.probes == [QUERY] and len(planner_probes.plans) == 1, (
        "a researcher without the attribute must plan exactly as today: one probe "
        f"search for the query and one planning call. measured probes "
        f"{planner_probes.probes} and {len(planner_probes.plans)} planner call(s) - "
        "zero of each means the AttributeError was swallowed by the gather's "
        "`except Exception: return []` rather than propagating"
    )
    assert researched == PLANNED + [QUERY], (
        "a researcher without the attribute must research the planned sub-queries plus "
        f"the original query, byte for byte as today. expected {PLANNED + [QUERY]}, "
        f"measured {researched}"
    )


# ---------------------------------------------------------------------------
# The tree side of P1.3
# ---------------------------------------------------------------------------
# Everything above drives the constructor and the conductor in isolation. That is
# necessary but not sufficient: an implementation can add the named parameter, add the
# conductor branch, go green on every test in this file — and never pass a preset from
# the only caller the saving is about. The tree would keep buying a probe search and a
# planner LLM call per node, which is 2 of the 8-9 sessions P1 exists to remove.
#
# So this pins the wiring: `research_node` hands each node its OWN question as the preset.

TREE_ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
TREE_CHILD_QS = ["How is outbox table growth bounded in production?",
                 "Which brokers deduplicate replayed outbox messages?"]
TREE_ANSWER = ("ANSWER: outbox answer body\nDIGEST: outbox digest\n"
               "LEARNINGS:\n- one claim\n")
TREE_DOC_URL = "https://example.test/outbox"


def _tree_cfg():
    return SimpleNamespace(
        fast_llm_provider="fake", fast_llm_model="fast-sentinel",
        strategic_llm_provider="fake", strategic_llm_model="strategic-sentinel",
        smart_llm_provider="fake", smart_llm_model="smart-sentinel",
        llm_kwargs={}, smart_retriever_config=None,
        smart_retriever_force_category=None, config_path=None)


@pytest.mark.asyncio
async def test_every_node_researcher_is_handed_its_own_question_as_the_preset():
    """P1.3 wired to its only caller: each node's researcher is built with
    `preset_sub_queries=[that node's question]`, so no node pays for planning."""
    import gpt_researcher.skills.tree_research as tr

    # research_node fails a node closed below MIN_CONTEXT_CHARS, and a FAILED root is
    # never expanded, so the stand-in context has to be a plausible size.
    context = ("The transactional outbox writes the message and the row in one commit. "
               * (tr.MIN_CONTEXT_CHARS // 40))
    built: list = []

    class _FakeNodeResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _tree_cfg()
            self.visited_urls = set()
            built.append(self)

        async def conduct_research(self):
            return context

        def get_research_sources(self):
            return [{"url": TREE_DOC_URL, "raw_content": context}]

        def get_costs(self):
            return 0.0

    emb = {TREE_ROOT_Q: [1.0, 0.0, 0.0, 0.0],
           TREE_CHILD_QS[0]: [0.0, 1.0, 0.0, 0.0],
           TREE_CHILD_QS[1]: [0.0, 0.0, 1.0, 0.0]}

    async def _embed(text):
        return list(emb.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def _children(node):
        return list(TREE_CHILD_QS) if node.depth == 0 else []

    parent = SimpleNamespace(query=TREE_ROOT_Q, cfg=_tree_cfg(), tone=None,
                             websocket=None, headers={}, visited_urls=set())
    skill = tr.TreeResearchSkill(parent)
    skill.embed_question = _embed

    from unittest import mock
    with mock.patch.object(tr, "GPTResearcher", _FakeNodeResearcher), \
         mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value=TREE_ANSWER)), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children):
        await skill.run(query=TREE_ROOT_Q, max_depth=1, max_breadth=2,
                        max_nodes=3, node_concurrency=3)

    node_builds = [r for r in built if r.query in [TREE_ROOT_Q] + TREE_CHILD_QS]
    assert len(node_builds) == 3, (
        "fixture precondition: one researcher per node for three nodes, got "
        f"{len(node_builds)}")

    missing = [r.query for r in node_builds
               if r.ctor_kwargs.get("preset_sub_queries") != [r.query]]
    assert not missing, (
        f"{len(missing)} of {len(node_builds)} node researchers were built without "
        "`preset_sub_queries=[<the node's own question>]`, so each of them still runs "
        "the planner: one probe search plus one planning LLM call per node, which is "
        "the 2-of-8 sessions P1.3 exists to remove. measured presets: "
        f"{[(r.query[:40], r.ctor_kwargs.get('preset_sub_queries')) for r in node_builds]}"
    )


@pytest.mark.asyncio
async def test_the_preset_widens_the_per_query_result_cap_it_replaces():
    """Bypassing the planner must not quietly cut the node's evidence by 4x.

    Measured 2026-09-05, live, the defect this pins. SmartRetriever fans out to 3-6
    retrievers (15-24 URLs for most categories) and then TRUNCATES to
    cfg.max_search_results_per_query, which defaults to 5. The planner path ran
    MAX_ITERATIONS + 1 = 4 sub-queries, so a node read up to 20 documents; the preset
    path runs ONE, so it read 5. Context per node fell from 45,712-60,118 chars across
    the 2026-09-02 production runs to 15,387 and 13,463 on the first slimmed run, and
    citations with it (9 against 20-26).

    Nothing caught it: 14k is far above MIN_CONTEXT_CHARS, so no node FAILED and the
    failed-ratio gate stayed green while the evidence behind every claim thinned out.

    The compensation costs no CLI session at all — the retrievers already returned those
    URLs and the cap is what throws them away. So a node researching one preset query
    must be allowed the documents the sub-queries it replaced would have been allowed.
    """
    import gpt_researcher.skills.tree_research as tr
    from unittest import mock

    context = ("The transactional outbox writes the message and the row in one commit. "
               * (tr.MIN_CONTEXT_CHARS // 40))
    built: list = []
    BASE_CAP = 5
    MAX_ITERATIONS = 3

    def _cfg():
        c = _tree_cfg()
        c.max_search_results_per_query = BASE_CAP
        c.max_iterations = MAX_ITERATIONS
        return c

    class _FakeNodeResearcher:
        def __init__(self, query=None, **kwargs):
            self.query = query
            self.ctor_kwargs = kwargs
            self.cfg = _cfg()
            self.visited_urls = set()
            built.append(self)

        async def conduct_research(self):
            return context

        def get_research_sources(self):
            return [{"url": TREE_DOC_URL, "raw_content": context}]

        def get_costs(self):
            return 0.0

    async def _embed(text):
        return [1.0, 0.0, 0.0, 0.0] if text == TREE_ROOT_Q else [0.0, 1.0, 0.0, 0.0]

    async def _children(node):
        return []

    parent = SimpleNamespace(query=TREE_ROOT_Q, cfg=_cfg(), tone=None,
                             websocket=None, headers={}, visited_urls=set())
    skill = tr.TreeResearchSkill(parent)
    skill.embed_question = _embed

    with mock.patch.object(tr, "GPTResearcher", _FakeNodeResearcher), \
         mock.patch.object(tr, "create_chat_completion",
                           new=mock.AsyncMock(return_value=TREE_ANSWER)), \
         mock.patch.object(skill, "generate_child_questions", side_effect=_children):
        await skill.run(query=TREE_ROOT_Q, max_depth=0, max_nodes=1)

    assert len(built) == 1, f"fixture precondition: one node researcher, got {len(built)}"
    caps = [r.cfg.max_search_results_per_query for r in built]
    expected = BASE_CAP * (MAX_ITERATIONS + 1)
    assert caps == [expected], (
        f"the node researching ONE preset query kept the per-sub-query cap {caps[0]}, so "
        f"it reads at most {caps[0]} documents where the {MAX_ITERATIONS + 1} sub-queries "
        f"it replaced would have read up to {expected}. That is the measured 4x context "
        f"collapse (45-60k chars before, 13-15k after) — and it costs nothing to avoid, "
        f"because the retrievers already returned those URLs and this cap is what "
        f"discards them"
    )
