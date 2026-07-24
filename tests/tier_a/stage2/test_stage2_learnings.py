"""Stage 2 RED tests — deep_research learnings compression relief.

Target: DeepResearchSkill.process_research_results must cap learnings at the
configured DEEP_RESEARCH_LEARNINGS (default 8, env-overridable) instead of the
old hardcoded 3, and pass DEEP_RESEARCH_LEARNINGS_TOKENS (default 2500) as
max_tokens instead of the old 1000. Citation/source parsing must not change.

Deterministic, no network: create_chat_completion is mocked with an AsyncMock.
"""
import asyncio
from unittest import mock

import pytest

from gpt_researcher.config import Config
from gpt_researcher.skills.deep_research import DeepResearchSkill

# URLs deliberately have no scheme colon in the [url]: style so the parser's
# split(':', 1) behavior stays identical before and after the change.
def _llm_response(n_learnings: int, n_questions: int = 10) -> str:
    lines = [
        f"Learning [src{i}.example.com/page]: Insight {i}" for i in range(1, n_learnings + 1)
    ]
    lines += [f"Question: Follow-up {i}?" for i in range(1, n_questions + 1)]
    return "\n".join(lines)


class _FakeResearcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.websocket = None
        self.tone = None
        self.headers = {}
        self.visited_urls = set()


def _process(cfg: Config, response_text: str):
    skill = DeepResearchSkill(_FakeResearcher(cfg))
    llm = mock.AsyncMock(return_value=response_text)
    with mock.patch("gpt_researcher.skills.deep_research.create_chat_completion", new=llm):
        result = asyncio.run(skill.process_research_results(query="q", context="ctx"))
    return result, llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DEEP_RESEARCH_LEARNINGS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_LEARNINGS_TOKENS", raising=False)


# ---------------------------------------------------------------------------
# (a) config default DEEP_RESEARCH_LEARNINGS=8: 10 learnings in -> 8 out, not 3
# ---------------------------------------------------------------------------

class TestLearningsCap:
    def test_config_defaults_exist(self):
        cfg = Config()
        assert getattr(cfg, "deep_research_learnings", None) == 8
        assert getattr(cfg, "deep_research_learnings_tokens", None) == 2500

    def test_default_returns_eight_learnings_not_three(self):
        result, _ = _process(Config(), _llm_response(10))
        assert result["learnings"] == [f"Insight {i}" for i in range(1, 9)], (
            "with 10 learnings from the LLM the configured default cap of 8 must apply, "
            "not the old hardcoded 3"
        )
        assert len(result["followUpQuestions"]) == 8

    def test_max_tokens_from_config(self):
        _, llm = _process(Config(), _llm_response(10))
        assert llm.call_args.kwargs.get("max_tokens") == 2500, (
            "max_tokens must come from DEEP_RESEARCH_LEARNINGS_TOKENS (default 2500), "
            "not the old hardcoded 1000"
        )


# ---------------------------------------------------------------------------
# (b) env override: DEEP_RESEARCH_LEARNINGS=5 -> 5 returned
# ---------------------------------------------------------------------------

class TestEnvOverride:
    def test_env_override_learnings(self, monkeypatch):
        monkeypatch.setenv("DEEP_RESEARCH_LEARNINGS", "5")
        result, _ = _process(Config(), _llm_response(10))
        assert result["learnings"] == [f"Insight {i}" for i in range(1, 6)]

    def test_env_override_tokens(self, monkeypatch):
        monkeypatch.setenv("DEEP_RESEARCH_LEARNINGS_TOKENS", "1234")
        _, llm = _process(Config(), _llm_response(10))
        assert llm.call_args.kwargs.get("max_tokens") == 1234


# ---------------------------------------------------------------------------
# (c) citation/source parsing unchanged under the new cap
# (asserts the new cap of 8 too, so it is RED before implementation)
# ---------------------------------------------------------------------------

class TestCitationParsingUnchanged:
    def test_shape_and_citations_unchanged(self):
        # 9 bracket-style learnings + 1 bare-URL learning = both parser branches
        response = "\n".join(
            [f"Learning [src{i}.example.com/page]: Insight {i}" for i in range(1, 10)]
            + ["Learning: The grid stores energy https://b.example.com/grid"]
            + ["Question: What next?"]
        )
        result, _ = _process(Config(), response)

        assert set(result) == {"learnings", "followUpQuestions", "citations"}
        # new cap of 8 applies, order preserved (old code returned only 3)
        assert result["learnings"] == [f"Insight {i}" for i in range(1, 9)]
        # bracket-style citations parse exactly as before, per learning
        for i in range(1, 9):
            assert result["citations"][f"Insight {i}"] == f"src{i}.example.com/page"
        # bare-URL branch still parses into the (uncapped) citations dict
        assert result["citations"]["The grid stores energy"] == "https://b.example.com/grid"
        assert result["followUpQuestions"] == ["What next?"]
