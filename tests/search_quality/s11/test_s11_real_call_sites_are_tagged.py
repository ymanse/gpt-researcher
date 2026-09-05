"""s11 / P0 rule 4b: the REAL call sites must be the ones that charge the site keys.

`test_s11_call_sites_are_attributed.py` charges `note_agent_call()` by hand inside an
`agent_purpose(...)` block, so everything it proves is that the counter works. An
implementation can ship `SITES`, `agent_purpose` and both delta functions, wrap ZERO
production call sites, go fully green on that file — and still report
`agent_calls_by_site() == {"untagged": N}` on every live run, which is the same blind
total `agent_calls_spent()` already gives. Worse, the gate "classify + choose_agent <= 2
per run" would then pass VACUOUSLY by reading 0 for both.

So this file drives each tagged site's real function with the LLM boundary faked, and
asserts the charge landed on that site's key. The fake stands exactly where
`chat_model._run_query` sits: it charges one `note_agent_call()` per invocation (one CLI
session per spawn) and returns a canned reply, so the tag has to come from the
production code between the entry point and that spawn — never from the test.

Every test asserts the drive FIRST (`spawn.calls == 1`). That separates the two ways
this file can go red: a fake that the real code never reached (a broken fixture) fails
on that line, while an unwrapped call site fails on the attribution line below it.

Sites covered here, one test each — `choose_agent`, `plan`, `classify`, `answer`,
`children`, `merge`. `verify` lives in gptr-mcp (`verification.py`), a different repo,
and is out of this suite's reach.

`classify` is asserted only for its SITE KEY. That the tag survives the retriever's
ThreadPoolExecutor hop is the sibling file's topology contract (P0 rule 1) and is not
re-litigated here.

Hermetic: no network, no LLM, no CLI subprocess, no embeddings service, and
`prune_cli_sessions` is stubbed so no test deletes transcripts under the operator's real
~/.claude. The budget is armed generously and DISARMED on teardown — s11 sorts before
s2-s10, and a leaked allowance silently converts every later tree run from unbounded to
bounded (spec gate table, "test hygiene").
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import gpt_researcher.actions.agent_creator as agent_creator
import gpt_researcher.actions.query_processing as query_processing
import gpt_researcher.skills.tree_research as tr
import gpt_researcher.utils.llm as llm_module
from gpt_researcher.llm_provider.claude_agent import _subscription as sub
from gpt_researcher.retrievers.smart.smart_retriever import ROUTING_TABLE, SmartRetriever

# Far above anything this file can spend: the budget's own arithmetic is
# tests/test_agent_budget_reserve.py's contract, not this file's.
ALLOWANCE = 1000

ROOT_Q = "What are the practical failure modes of the transactional outbox pattern?"
NODE_Q = "How is outbox table growth bounded in production?"

FAST_MODEL = "fast-classifier-sentinel"
STRATEGIC_MODEL = "strategic-writer-sentinel"
SMART_MODEL = "smart-agent-sentinel"

CATEGORY = "code_technical"
assert CATEGORY in ROUTING_TABLE

DOC_URL = "https://example.test/outbox"
# research_node fails a node closed below MIN_CONTEXT_CHARS, and a FAILED node never
# reaches the answer LLM cleanly, so the stand-in context has to be a plausible size.
CONTEXT = ("The transactional outbox writes the message row and the business row in one "
           "commit, and a relay publishes it later. ") * (tr.MIN_CONTEXT_CHARS // 40)

ANSWER_BLOB = ("ANSWER: The relay is the failure surface.\n"
               "DIGEST: The relay is the failure surface.\n"
               "LEARNINGS:\n- A stalled relay silently stops publication.\n")

CHILD_BLOB = ("Question: What does the originating vendor's own documentation say?\n"
              "Question: How do brokers deduplicate replayed outbox messages?\n")

CLAIM_A = "A stalled outbox relay stops publication without raising an error."
CLAIM_B = "Outbox table growth is bounded by deleting rows the relay has published."


@pytest.fixture(autouse=True)
def armed_run(monkeypatch):
    """Fresh baseline per test, no housekeeping on the operator's real home, and the
    allowance handed back so no later stage inherits a ceiling from s11."""
    monkeypatch.setattr(sub, "prune_cli_sessions", lambda *a, **k: 0)
    sub.begin_agent_run(ALLOWANCE)
    yield
    sub.begin_agent_run(0)


class _Spawn:
    """Stands in for `create_chat_completion` at the seam `chat_model` charges.

    One invocation == one `claude` CLI session == one `note_agent_call()`. The reply is
    canned; the SITE has to come from the production code that made the call.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        sub.note_agent_call()
        return self.reply


def _cfg(**over):
    """The config shape these call sites read — all three LLM tiers, because the sites
    do not agree on which one they ask."""
    base = dict(fast_llm_provider="fake", fast_llm_model=FAST_MODEL,
                strategic_llm_provider="fake", strategic_llm_model=STRATEGIC_MODEL,
                smart_llm_provider="fake", smart_llm_model=SMART_MODEL,
                llm_kwargs={}, max_iterations=3, temperature=0.4,
                strategic_token_limit=4000, smart_token_limit=4000,
                smart_retriever_config=None, smart_retriever_force_category=None,
                config_path=None)
    base.update(over)
    return SimpleNamespace(**base)


def _skill() -> tr.TreeResearchSkill:
    parent = SimpleNamespace(query=ROOT_Q, cfg=_cfg(), tone=None, websocket=None,
                             headers={}, visited_urls=set())
    return tr.TreeResearchSkill(parent)


def _charged(site: str, spawn: _Spawn) -> str:
    """The message a mis-attributed site has to explain."""
    by_site = sub.agent_calls_by_site()
    return (f"{spawn.calls} CLI session(s) were spawned by the real call site, and "
            f"{by_site.get(site, 0)} of them were charged to {site!r}; full breakdown "
            f"{by_site}. The site is not wrapped in agent_purpose({site!r}), so in "
            f"production its spend reports as 'untagged' and the P1 saving at this site "
            f"cannot be measured before or after.")


async def test_choosing_the_agent_charges_the_choose_agent_site():
    """`choose_agent` is one CLI session per node researcher — P1.2 exists to resolve it
    once per run, which is unmeasurable until the site names itself."""
    spawn = _Spawn('{"server": "Outbox Agent", "agent_role_prompt": "You research outboxes."}')

    with mock.patch.object(agent_creator, "create_chat_completion", new=spawn):
        agent, role = await agent_creator.choose_agent(query=NODE_Q, cfg=_cfg())

    assert spawn.calls == 1 and agent == "Outbox Agent" and role, (
        f"the fixture never drove the real agent selection: {spawn.calls} LLM call(s), "
        f"agent={agent!r} — nothing about attribution is being measured")
    assert sub.agent_calls_by_site().get("choose_agent", 0) == 1, _charged("choose_agent", spawn)


async def test_planning_the_outline_charges_the_plan_site():
    """`plan_research_outline` is the sub-query planner P1.3 bypasses with a preset; the
    saving is only visible if its own session is attributed to `plan`."""
    spawn = _Spawn('["outbox relay failure modes", "outbox table growth bounds"]')

    with mock.patch.object(query_processing, "create_chat_completion", new=spawn):
        sub_queries = await query_processing.plan_research_outline(
            query=NODE_Q, search_results=[], agent_role_prompt="You research outboxes.",
            cfg=_cfg(), parent_query=ROOT_Q, report_type="research_report",
            retriever_names=[])

    assert spawn.calls == 1 and sub_queries == ["outbox relay failure modes",
                                                "outbox table growth bounds"], (
        f"the fixture never drove the real planner: {spawn.calls} LLM call(s) returning "
        f"{sub_queries!r} — nothing about attribution is being measured")
    assert sub.agent_calls_by_site().get("plan", 0) == 1, _charged("plan", spawn)


def test_classifying_a_query_charges_the_classify_site():
    """`_classify_query` is the site P1.1 collapses to one call per run. Driven from a
    plain sync caller (no running loop), because whether the tag survives the retriever's
    thread hop is the sibling file's topology contract, not this one's."""
    spawn = _Spawn(CATEGORY)

    with mock.patch.object(llm_module, "create_chat_completion", new=spawn):
        category = SmartRetriever(NODE_Q, cfg=_cfg())._classify_query()

    assert spawn.calls == 1 and category == CATEGORY, (
        f"the fixture never drove the real classifier cleanly: {spawn.calls} LLM call(s) "
        f"returning {category!r} (`_classify_query` swallows its own exceptions and "
        "answers 'general_web') — nothing about attribution is being measured")
    assert sub.agent_calls_by_site().get("classify", 0) == 1, _charged("classify", spawn)


async def test_answering_a_node_charges_the_answer_site():
    """`research_node`'s ANSWER/DIGEST/LEARNINGS call is the one session per node that
    buys the research; it must be separable from the overhead around it, or `calls per
    researched node <= 3` cannot be read."""
    spawn = _Spawn(ANSWER_BLOB)

    class _FakeNodeResearcher:
        # **kwargs: research_node already passes 8 keywords and P1.2/P1.3 add more
        def __init__(self, query=None, **kwargs):
            self.query = query
            # P1.1 stamps `researcher.cfg.smart_retriever_force_category` on the node
            # researcher AFTER construction, inside this very method — a fake without a
            # cfg would die of AttributeError the day P1.1 lands and read as a broken
            # fixture rather than an unwrapped site.
            self.cfg = _cfg()
            self.visited_urls = set()

        async def conduct_research(self):
            return CONTEXT

        def get_research_sources(self):
            return [{"url": DOC_URL, "raw_content": CONTEXT}]

        def get_costs(self):
            return 0.0

    skill = _skill()
    node = tr.ResearchNode(id="0", question=NODE_Q, parent_id=None, depth=0)
    with mock.patch.object(tr, "GPTResearcher", _FakeNodeResearcher), \
         mock.patch.object(tr, "create_chat_completion", new=spawn):
        await skill.research_node(node)

    assert spawn.calls == 1 and node.status is tr.NodeStatus.ANSWERED, (
        f"the fixture never drove the real answer pass: {spawn.calls} LLM call(s), node "
        f"status {node.status} — research_node fails closed before the answer LLM when "
        "the node read no document, and a node that never answered charges nothing")
    assert sub.agent_calls_by_site().get("answer", 0) == 1, _charged("answer", spawn)


async def test_generating_child_questions_charges_the_children_site():
    """`generate_child_questions` is the ONE call P1.4's gate exists to skip. If it is
    not attributed, a run cannot tell an expansion it paid for from one it did not."""
    spawn = _Spawn(CHILD_BLOB)

    skill = _skill()
    node = tr.ResearchNode(id="0", question=NODE_Q, parent_id=None, depth=0,
                           answer_digest="A stalled relay stops publication.")
    with mock.patch.object(tr, "create_chat_completion", new=spawn):
        questions = await skill.generate_child_questions(node)

    assert spawn.calls == 1 and len(questions) == 2, (
        f"the fixture never drove the real expansion: {spawn.calls} LLM call(s) yielding "
        f"{questions!r} — nothing about attribution is being measured")
    assert sub.agent_calls_by_site().get("children", 0) == 1, _charged("children", spawn)


async def test_the_claim_equivalence_judge_charges_the_merge_site():
    """The roll-up's judge (`_covered`) runs AFTER expansion has spent the run down, and
    `agent_synthesis_reserve()` exists to keep calls back for it. Attributing it is how a
    reserve that is too small or too large gets measured instead of guessed."""
    spawn = _Spawn(f'"{CLAIM_A[:40]}" => UNIQUE: none\n"{CLAIM_B[:40]}" => UNIQUE: the bound\n')

    skill = _skill()
    with mock.patch.object(tr, "create_chat_completion", new=spawn):
        await skill._covered([CLAIM_A, CLAIM_B])

    assert spawn.calls == 1, (
        f"the fixture never drove the real merge judge: {spawn.calls} LLM call(s) — "
        "`_covered` returns early with no model, fewer than two units, or a cleared "
        "`_merge_judge_ok`, and an early return charges nothing")
    assert sub.agent_calls_by_site().get("merge", 0) == 1, _charged("merge", spawn)
