"""Stage 5 RED tests — scope/clarification gating of the auto-proceed answer.

Target: DeepResearchSkill.run() gains a scope: bool keyword (default False,
threaded through from the MCP deep_research tool).

Contract pinned here (GREEN must satisfy exactly this):
- run(on_progress=None, scope=False); the skill always exposes .scope_brief
  (None until a scope=True run builds it).
- scope=False (and no scope arg at all): today's behavior — every
  clarification question auto-answered "Automatically proceeding with
  research", no brief built, no direct LLM call from run().
- scope=True: a 1-round brief dict {"query", "questions", "scope_statement"}
  built from the original query, the generate_research_plan clarification
  questions, and a create_chat_completion call for the scope statement;
  deep_research then receives a query containing the scope statement and
  NOT the auto-answer text.

Deterministic, no network: create_chat_completion, generate_research_plan,
deep_research and CitationAgent are all mocked.
"""
import asyncio
import inspect
from unittest import mock

from gpt_researcher.config import Config
from gpt_researcher.skills.deep_research import DeepResearchSkill

AUTO_ANSWER = "Automatically proceeding with research"
QUESTIONS = ["What time frame matters?", "Which region should be covered?"]
SCOPE_STATEMENT = "Scope: grid-scale storage economics in Europe, 2025-2026."

DEEP_RESULTS = {
    "learnings": ["Insight one"],
    "visited_urls": ["https://a.example.com/page"],
    "citations": {"Insight one": "https://a.example.com/page"},
    "context": ["context body"],
    "sources": [],
}


class _FakeResearcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.query = "original query about grid storage"
        self.websocket = None
        self.tone = None
        self.headers = {}
        self.visited_urls = set()
        self.log_handler = None

    def get_costs(self) -> float:
        return 0.0


def _run(**run_kwargs):
    """Run the skill with everything below run() mocked; return (skill, dr, llm)."""
    skill = DeepResearchSkill(_FakeResearcher(Config()))
    llm = mock.AsyncMock(return_value=SCOPE_STATEMENT)
    dr = mock.AsyncMock(return_value=dict(DEEP_RESULTS))
    plan = mock.AsyncMock(return_value=list(QUESTIONS))
    ca = mock.MagicMock()
    ca.return_value.verify.return_value = {
        "total_claims": 0, "grounded": 0, "unverified": 0,
    }
    with mock.patch(
        "gpt_researcher.skills.deep_research.create_chat_completion", new=llm
    ), mock.patch(
        "gpt_researcher.skills.citation_verification.CitationAgent", new=ca
    ), mock.patch.object(skill, "generate_research_plan", new=plan), \
            mock.patch.object(skill, "deep_research", new=dr):
        asyncio.run(skill.run(**run_kwargs))
    return skill, dr, llm


def _research_query(dr: mock.AsyncMock) -> str:
    call = dr.call_args
    assert call is not None, "deep_research was never called — research must proceed"
    return call.kwargs.get("query") or call.args[0]


# ---------------------------------------------------------------------------
# (a) scope=False -> auto-proceed exactly as today, no brief, no behavior change
# ---------------------------------------------------------------------------

class TestScopeOff:
    def test_run_signature_has_scope_default_false(self):
        params = inspect.signature(DeepResearchSkill.run).parameters
        assert "scope" in params, "run() must accept a scope keyword"
        assert params["scope"].default is False, "scope must default to False"

    def test_scope_false_auto_proceeds_with_no_brief(self):
        skill, dr, llm = _run(scope=False)
        query = _research_query(dr)
        assert AUTO_ANSWER in query, "scope=False must keep the auto-answer flow"
        for q in QUESTIONS:
            assert q in query, "clarification questions must still reach research"
        assert skill.scope_brief is None, "scope=False must not build a brief"
        assert llm.await_count == 0, (
            "scope=False must not add any LLM call in run() (regression guard)"
        )

    def test_default_call_without_scope_arg_unchanged(self):
        skill, dr, llm = _run()
        assert AUTO_ANSWER in _research_query(dr)
        assert skill.scope_brief is None
        assert llm.await_count == 0


# ---------------------------------------------------------------------------
# (b) scope=True -> non-empty 1-round brief, research proceeds with it
# ---------------------------------------------------------------------------

class TestScopeOn:
    def test_scope_true_builds_nonempty_brief(self):
        skill, _, _ = _run(scope=True)
        brief = skill.scope_brief
        assert brief, "scope=True must produce a non-empty brief object"
        assert brief["query"] == "original query about grid storage"
        assert brief["questions"] == QUESTIONS
        assert brief["scope_statement"].strip(), "scope statement must be non-empty"
        assert SCOPE_STATEMENT in brief["scope_statement"], (
            "scope statement must come from the (mocked) LLM"
        )

    def test_scope_true_research_proceeds_with_brief(self):
        _, dr, _ = _run(scope=True)
        assert dr.await_count == 1, "research must still run exactly once"
        query = _research_query(dr)
        assert SCOPE_STATEMENT in query, (
            "the confirmed scope must flow into the research query"
        )
        assert AUTO_ANSWER not in query, (
            "scope=True replaces the auto-answer, it must not leak through"
        )

    def test_scope_true_uses_llm_for_scope_statement(self):
        _, _, llm = _run(scope=True)
        assert llm.await_count >= 1, (
            "the 1-round brief needs an LLM call to confirm the scope"
        )
