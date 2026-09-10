"""The claim-equivalence judge reads MERGE_LLM, and inherits STRATEGIC when it is unset.

The judge was welded to `strategic_llm_*`, which is also what node ANSWERS use. Those
two sites want different things: an answer is open-ended synthesis over retrieved
sources, while the judge is handed two statements the encoder already paired and asked
whether one says anything the other does not. Sharing a role meant the only way to make
the judge cheaper was to make the answers cheaper too.

Splitting the role is only safe because of which way the judge fails. `_covered` returns
`{}` on every unclear outcome -- no model, a timeout, a reply naming no unit -- and a
claim that fails to merge costs report LENGTH (`synthesis_ratio_pct_max`) while a claim
wrongly merged costs a FACT. A weaker judge therefore produces a longer report, not a
less accurate one, which is the trade a deployment should be allowed to make.

What must not change is the default: an unset MERGE_LLM has to behave exactly as before,
or this becomes a silent quality change for every existing deployment.
"""
from types import SimpleNamespace

from gpt_researcher.skills import tree_research as tr


def _skill(**cfg_over):
    """A TreeResearchSkill whose researcher carries only the config the model
    resolution reads. Built through __new__: __init__ reaches for a live researcher,
    and the seam under test is the role resolution, not the construction."""
    base = dict(strategic_llm_provider="claude_agent", strategic_llm_model="sonnet")
    base.update(cfg_over)
    skill = tr.TreeResearchSkill.__new__(tr.TreeResearchSkill)
    skill.researcher = SimpleNamespace(cfg=SimpleNamespace(**base))
    skill._strategic_llm = None
    skill._merge_llm = None
    return skill


def test_an_unset_merge_llm_inherits_the_strategic_model():
    """The default. Every deployment that predates MERGE_LLM must be unaffected."""
    skill = _skill(merge_llm_provider=None, merge_llm_model=None)

    assert skill._merge_model() == ("claude_agent", "sonnet"), (
        f"an unset MERGE_LLM resolved to {skill._merge_model()}, not the strategic "
        "model -- adding the key must not change what existing deployments run")


def test_a_set_merge_llm_is_used_for_the_judge():
    """The point of the key: the judge moves without the answers moving."""
    skill = _skill(merge_llm_provider="openrouter",
                   merge_llm_model="deepseek/deepseek-v4-pro")

    assert skill._merge_model() == ("openrouter", "deepseek/deepseek-v4-pro"), (
        f"MERGE_LLM was set but the judge resolved {skill._merge_model()}")
    assert skill._strategic_model() == ("claude_agent", "sonnet"), (
        f"the strategic model moved to {skill._strategic_model()} -- node answers must "
        "keep the reasoning tier, or this stops being a merge-only downgrade")


def test_a_half_set_merge_llm_falls_back_rather_than_resolving_half_a_model():
    """A provider with no model (or the reverse) is a misconfiguration. Resolving it
    half-way would hand `create_chat_completion` a None and kill the judge for the run;
    inheriting is the outcome that still merges."""
    for over in ({"merge_llm_provider": "openrouter", "merge_llm_model": None},
                 {"merge_llm_provider": None, "merge_llm_model": "some/model"}):
        skill = _skill(**over)
        assert skill._merge_model() == ("claude_agent", "sonnet"), (
            f"half-set {over} resolved to {skill._merge_model()} instead of falling "
            "back to the strategic model")


def test_the_resolved_role_is_cached_for_the_assembly():
    """`_covered` is called once per candidate screen -- 52-350 claim units on a real
    tree -- and re-resolving would rebuild a live Config on each one."""
    skill = _skill(merge_llm_provider="openrouter", merge_llm_model="x/y")
    first = skill._merge_model()
    skill.researcher.cfg.merge_llm_model = "changed/after/resolution"

    assert skill._merge_model() == first, (
        "the merge model was re-resolved mid-assembly; it is cached per run so the "
        "judge cannot change model between two screens of the same roll-up")


def test_merge_llm_is_a_declared_config_key_defaulting_to_inherit():
    """The key has to exist in the schema and default to empty -- config.py guards on
    exactly that, because parse_llm('') raises rather than returning (None, None)."""
    from gpt_researcher.config.variables.base import BaseConfig
    from gpt_researcher.config.variables.default import DEFAULT_CONFIG

    assert "MERGE_LLM" in BaseConfig.__annotations__, (
        "MERGE_LLM is missing from BaseConfig, so a deployment setting it gets no "
        "type and no documentation")
    assert DEFAULT_CONFIG["MERGE_LLM"] == "", (
        f"MERGE_LLM defaults to {DEFAULT_CONFIG['MERGE_LLM']!r}; it must default to "
        "empty, which is what means 'inherit STRATEGIC_LLM'")
