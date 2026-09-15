"""s15: the linear deep path and the report writer must say where their sessions went.

Measured 2026-09-15 on a real `deep_research` call (528s, 140 sources):

    agent_calls_by_site = {"classify": 2, "untagged": 14, "choose_agent": 1, "plan": 9}

Fourteen of twenty-six CLI sessions -- more than half -- charged to nobody. The tree path
was instrumented in s11; `report_type="deep"` routes through `DeepResearchSkill`, whose
own query-generation, question-generation and learnings calls were never wrapped, and
`write_report` spends its sessions in `actions/report_generation`, also unwrapped. A cost
breakdown that cannot see half the cost is not a breakdown.

Sites are named for the KIND of work so the two modes stay comparable
(see `utils/agent_purpose.SITES`): the linear skill's query generation is `plan`, its
follow-up questions are `children`, its learnings extraction is `answer`. Only the scope
brief and report writing have no tree counterpart and get names of their own.

Two kinds of test, for the reason s11 gave: a test that charges the counter by hand proves
only that the counter works. The behavioural tests drive each REAL function with the LLM
boundary faked, so the tag has to come from production code. The static test covers what
is too heavy to drive (`run(scope=True)` is a whole research run) and pins the fallback
tiers, each of which spawns its own session.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.actions.report_generation as report_generation
import gpt_researcher.skills.deep_research as deep_research
from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.utils.enum import Tone

ALLOWANCE = 1000
QUERY = "What are the practical failure modes of the transactional outbox pattern?"
CONTEXT = "The outbox writes the message row and the business row in one commit. " * 20


@pytest.fixture(autouse=True)
def armed_run(monkeypatch):
    """Fresh baseline per test, no pruning of the operator's real ~/.claude, and the
    allowance handed back so no later stage inherits a ceiling (see s11)."""
    monkeypatch.setattr(sub, "prune_cli_sessions", lambda *a, **k: 0)
    sub.begin_agent_run(ALLOWANCE)
    yield
    sub.begin_agent_run(0)


class _Spawn:
    """One invocation == one `claude` CLI session == one `note_agent_call()`.

    `fail_first` makes the first spawn raise AFTER charging -- a session that errors has
    still been spawned -- so a fallback tier can be driven."""

    def __init__(self, reply: str, fail_first: bool = False) -> None:
        self.reply, self.fail_first, self.calls = reply, fail_first, 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        sub.note_agent_call()
        if self.fail_first and self.calls == 1:
            raise RuntimeError("first tier refused")
        return self.reply


def _cfg():
    return SimpleNamespace(
        strategic_llm_provider="fake", strategic_llm_model="strategic-sentinel",
        smart_llm_provider="fake", smart_llm_model="smart-sentinel",
        fast_llm_provider="fake", fast_llm_model="fast-sentinel",
        llm_kwargs={}, smart_token_limit=4000, strategic_token_limit=4000,
        reasoning_effort="medium", temperature=0.4, language="english",
        total_words=1000, report_format="apa", config_path=None,
        deep_research_breadth=2, deep_research_depth=1, deep_research_concurrency=1)


def _skill() -> deep_research.DeepResearchSkill:
    researcher = SimpleNamespace(cfg=_cfg(), websocket=None, tone=None, headers={},
                                 visited_urls=set(), retrievers=[], query=QUERY)
    return deep_research.DeepResearchSkill(researcher)


def _charged(site: str, spawn: _Spawn) -> str:
    by_site = sub.agent_calls_by_site()
    return (f"{spawn.calls} CLI session(s) were spawned by the real call site and "
            f"{by_site.get(site, 0)} were charged to {site!r}; full breakdown {by_site}. "
            f"Unwrapped, this site reports as 'untagged' -- the 14-of-26 blind spot "
            f"measured on 2026-09-15")


# -- the linear skill ---------------------------------------------------------------

async def test_generating_serp_queries_charges_plan():
    spawn = _Spawn("Query: outbox relay stalls\nGoal: find failure modes\n")
    with mock.patch.object(deep_research, "create_chat_completion", new=spawn):
        queries = await _skill().generate_search_queries(QUERY, num_queries=1)

    assert spawn.calls == 1 and queries and queries[0]["query"] == "outbox relay stalls", (
        f"the fixture never drove the real query generator: {spawn.calls} call(s), "
        f"{queries!r} -- nothing about attribution is being measured")
    assert sub.agent_calls_by_site().get("plan", 0) == 1, _charged("plan", spawn)


async def test_generating_the_research_plan_charges_children():
    """`retrievers=[]` so the initial search is skipped: this test is about the LLM call's
    tag, and a search would reach `classify`, which s11 already owns."""
    spawn = _Spawn("Question: How do brokers deduplicate replayed messages?\n")
    with mock.patch.object(deep_research, "create_chat_completion", new=spawn):
        questions = await _skill().generate_research_plan(QUERY, num_questions=1)

    assert spawn.calls == 1 and questions == ["How do brokers deduplicate replayed messages?"], (
        f"the fixture never drove the real planner: {spawn.calls} call(s), {questions!r}")
    assert sub.agent_calls_by_site().get("children", 0) == 1, _charged("children", spawn)


async def test_extracting_learnings_charges_answer():
    spawn = _Spawn("Learning [https://example.test/o]: a stalled relay stops publication\n"
                   "Question: What bounds outbox table growth?\n")
    with mock.patch.object(deep_research, "create_chat_completion", new=spawn):
        result = await _skill().process_research_results(QUERY, CONTEXT, num_learnings=1)

    assert spawn.calls == 1 and result.get("learnings"), (
        f"the fixture never drove the real learnings extractor: {spawn.calls} call(s), "
        f"{result!r}")
    assert sub.agent_calls_by_site().get("answer", 0) == 1, _charged("answer", spawn)


# -- the report writer --------------------------------------------------------------

async def test_writing_the_introduction_charges_report():
    spawn = _Spawn("## Introduction\nThe outbox pattern ...")
    with mock.patch.object(report_generation, "create_chat_completion", new=spawn):
        intro = await report_generation.write_report_introduction(
            query=QUERY, context=CONTEXT, agent_role_prompt="You research outboxes.",
            config=_cfg())

    assert spawn.calls == 1 and intro.startswith("## Introduction"), (
        f"the fixture never drove the real introduction writer: {spawn.calls} call(s)")
    assert sub.agent_calls_by_site().get("report", 0) == 1, _charged("report", spawn)


async def test_the_report_fallback_tier_is_charged_to_report_too():
    """`generate_report` retries with the role folded into the user turn when the first
    call raises. Both attempts are sessions, so both belong to `report` -- the same rule
    `plan` already follows for its three tiers."""
    spawn = _Spawn("# Report\nbody", fail_first=True)
    with mock.patch.object(report_generation, "create_chat_completion", new=spawn):
        report = await report_generation.generate_report(
            query=QUERY, context=CONTEXT, agent_role_prompt="You research outboxes.",
            report_type="research_report", tone=Tone.Objective, report_source="web",
            websocket=None, cfg=_cfg())

    assert spawn.calls == 2 and report == "# Report\nbody", (
        f"the fixture never reached the fallback tier: {spawn.calls} call(s), {report!r}")
    assert sub.agent_calls_by_site().get("report", 0) == 2, _charged("report", spawn)
    assert sub.agent_calls_by_site().get("untagged", 0) == 0, _charged("untagged", spawn)


# -- coverage: every call in both files, including the ones too heavy to drive -------

EXPECTED = {
    deep_research: {"generate_search_queries": "plan",
                    "generate_research_plan": "children",
                    "process_research_results": "answer",
                    "run": "scope"},
    report_generation: {"write_report_introduction": "report",
                        "write_conclusion": "report",
                        "summarize_url": "report",
                        "generate_draft_section_titles": "report",
                        "generate_report": "report"},
}


def _site_of(with_node: ast.With):
    for item in with_node.items:
        call = item.context_expr
        if (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "agent_purpose"
                and call.args and isinstance(call.args[0], ast.Constant)):
            return call.args[0].value
    return None


@pytest.mark.parametrize("module", list(EXPECTED), ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_llm_call_in_the_file_sits_under_its_site(module):
    tree = ast.parse(inspect.getsource(module))
    found, wrong = {}, []

    def visit(node, fn, site):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn, site = node.name, None
        if isinstance(node, ast.With):
            site = _site_of(node) or site
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "create_chat_completion"):
            found.setdefault(fn, []).append(site)
            if site != EXPECTED[module].get(fn):
                wrong.append(f"{fn}:{node.lineno} charged to {site!r}, "
                             f"expected {EXPECTED[module].get(fn)!r}")
        for child in ast.iter_child_nodes(node):
            visit(child, fn, site)

    visit(tree, None, None)
    assert not wrong, (
        f"{module.__name__}: LLM calls outside their agent_purpose block -- each one "
        f"reports as 'untagged' in production:\n  " + "\n  ".join(wrong))
    assert set(found) == set(EXPECTED[module]), (
        f"{module.__name__}: the functions that call the LLM changed "
        f"(now {sorted(found)}, expected {sorted(EXPECTED[module])}). A new call site "
        f"needs a site name here and in utils/agent_purpose.SITES")
