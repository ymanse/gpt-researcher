"""create_chat_completion must not spend a full 10-attempt exponential backoff on an
error that cannot possibly clear on retry.

Measured live (harness-search/no_read/tmp/probe_node_timing_result.json): one
`conduct_research()` call spent 395.19s wall, node FAILED, zero usable context. All 50
LLM calls failed with the identical error — the claude_agent CLI reporting a spent
weekly quota — across 5 `create_chat_completion()` invocations, each independently
burning a full 10-attempt exponential backoff (1,2,4,8,8,8,8,8,8s = 55s of pure
asyncio.sleep) before giving up. 271 of the 395s (69%) was sleep between doomed
retries; only 109.1s was spent actually calling out.

The contract this file pins: `gpt_researcher.utils.llm.is_llm_retryable_error`
classifies the CLI refusal (see `_CLI_REFUSAL_RE` in
gpt_researcher/llm_provider/claude_agent/chat_model.py, which raises the RuntimeError
this test reproduces) and HTTP 401/403 as NOT retryable — one attempt, no sleep — while
everything else (a transient network blip, a 502, an unrecognized exception) keeps
today's full 10-attempt retry behavior, unreduced.

Deterministic: no network, no real LLM call — `gpt_researcher.utils.llm.get_llm` is
replaced with a fake provider whose `get_chat_response` always raises, and
`asyncio.sleep` is stubbed to make the retryable-path test fast instead of ~55s slow.
"""
import pytest

import gpt_researcher.utils.llm as llm_mod
from gpt_researcher.utils.llm import create_chat_completion, is_llm_retryable_error


def _fake_provider(exc):
    """A GenericLLMProvider stand-in whose get_chat_response always raises `exc`."""
    state = {"calls": 0}

    class _Provider:
        async def get_chat_response(self, messages, stream, websocket=None, **kwargs):
            state["calls"] += 1
            raise exc

    return _Provider(), state


def _no_sleep(monkeypatch):
    async def _instant(*_a, **_k):
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _instant)


# ---------------------------------------------------------------------------
# is_llm_retryable_error: the classification itself
# ---------------------------------------------------------------------------

def test_cli_refusal_is_not_retryable():
    exc = RuntimeError(
        "[claude_agent] the CLI refused the request instead of answering: "
        "You've hit your weekly limit · resets 9pm (UTC)"
    )
    assert is_llm_retryable_error(exc) is False


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retryable(status):
    class _Auth(Exception):
        status_code = status

    assert is_llm_retryable_error(_Auth("nope")) is False


def test_status_less_and_gateway_errors_are_retryable():
    assert is_llm_retryable_error(RuntimeError("connection refused")) is True

    class _Gateway(Exception):
        status_code = 502

    assert is_llm_retryable_error(_Gateway("bad gateway")) is True


# ---------------------------------------------------------------------------
# create_chat_completion wiring: the outcome that actually matters
# ---------------------------------------------------------------------------

async def test_a_cli_refusal_is_attempted_once(monkeypatch):
    exc = RuntimeError(
        "[claude_agent] the CLI refused the request instead of answering: "
        "You've hit your weekly limit · resets 9pm (UTC)"
    )
    provider, state = _fake_provider(exc)
    monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: provider)
    _no_sleep(monkeypatch)

    with pytest.raises(RuntimeError):
        await create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="m", llm_provider="claude_agent",
        )

    assert state["calls"] == 1, (
        f"a refusal banner must fail fast — no retry, since the identical request "
        f"meets the identical exhausted quota. got {state['calls']} calls"
    )


async def test_a_transient_error_still_gets_all_ten_attempts(monkeypatch):
    provider, state = _fake_provider(RuntimeError("connection reset"))
    monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: provider)
    _no_sleep(monkeypatch)

    with pytest.raises(RuntimeError):
        await create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="m", llm_provider="openai",
        )

    assert state["calls"] == 10, (
        f"a transient error's retry count must NOT be reduced by this change. "
        f"got {state['calls']} calls"
    )


async def test_non_vacuous_gate_disabled_would_retry_the_refusal_too(monkeypatch):
    """Proves the two tests above actually exercise the gate: with the gate forced
    to always say 'retryable', the CLI refusal consumes all 10 attempts too — the
    same shape as today's pre-fix bug. If this didn't hold, test_a_cli_refusal_
    is_attempted_once could be passing for an unrelated reason."""
    exc = RuntimeError(
        "[claude_agent] the CLI refused the request instead of answering: "
        "You've hit your weekly limit · resets 9pm (UTC)"
    )
    provider, state = _fake_provider(exc)
    monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: provider)
    monkeypatch.setattr(llm_mod, "is_llm_retryable_error", lambda _exc: True)
    _no_sleep(monkeypatch)

    with pytest.raises(RuntimeError):
        await create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="m", llm_provider="claude_agent",
        )

    assert state["calls"] == 10, (
        "with the gate disabled the refusal must burn all attempts, exactly like "
        "the measured pre-fix bug — otherwise the gate isn't what's under test"
    )
