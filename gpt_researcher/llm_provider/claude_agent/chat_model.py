"""LangChain ``BaseChatModel`` backed by the Claude Agent SDK.

Routes gpt-researcher's LLM roles (FAST/SMART/STRATEGIC) through the
``claude_agent_sdk.query()`` single-turn call, which authenticates with the
user's Claude subscription via the ``claude`` CLI (see ``_subscription``).
This is **async-only** — gpt-researcher always invokes LLMs via ``ainvoke`` /
``astream``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from gpt_researcher.llm_provider.claude_agent._subscription import (
    _get_cli_env,
    _resolve_cli_path,
    get_concurrency_semaphore,
    note_agent_call,
)
from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

logger = logging.getLogger(__name__)

# What the `claude` CLI prints INSTEAD of an answer when it will not serve the request.
# These arrive as ordinary stdout content, so without this they are indistinguishable
# from a model reply — see the check at the end of _run_query for what that cost.
_CLI_REFUSAL_RE = re.compile(
    r"you'?ve hit your (?:weekly|usage|\w+) limit"
    r"|resets \w+ \d+, \d+(?::\d+)?\s*(?:am|pm)"
    r"|(?:please )?run\s+`?/?login"
    r"|invalid api key|authentication[ _]error|not logged in|credit balance is too low",
    re.I,
)


def _accepted_option_fields() -> set[str]:
    """Field names accepted by the installed ``ClaudeAgentOptions``.

    The SDK's option schema drifts across versions; we only pass kwargs the
    installed version actually declares so a newer/older SDK does not raise.
    """
    try:
        return {f.name for f in dataclasses.fields(ClaudeAgentOptions)}
    except TypeError:
        return set(inspect.signature(ClaudeAgentOptions).parameters)


class ChatClaudeAgent(BaseChatModel):
    """Claude subscription chat model (Agent SDK, single-turn).

    Each generation spawns a stateless ``query()`` call. No tools, no MCP, no
    filesystem settings — pure text generation through the subscription session.
    """

    model: str
    timeout_seconds: int = 300
    max_output_turns: int = 1

    model_config = {"extra": "ignore"}

    @property
    def _llm_type(self) -> str:
        return "claude_agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "timeout_seconds": self.timeout_seconds}

    # ── Sync path is unsupported (project is async-only) ──────────────
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "ChatClaudeAgent is async-only; use ainvoke()/astream()."
        )

    # ── Message → SDK input conversion ────────────────────────────────
    def _to_sdk_inputs(self, messages: Sequence[BaseMessage]) -> tuple[str, str]:
        """Split LangChain messages into ``(system_prompt, prompt)``.

        ``SystemMessage`` content feeds the system prompt. The common
        ``[system, user]`` shape collapses to the bare user string; anything
        richer is flattened into a ``"User: …\\n\\nAssistant: …"`` transcript.
        Content blocks are normalized with the same logic every other
        gpt-researcher response flows through.
        """
        normalize = GenericLLMProvider._normalize_content
        system_parts: list[str] = []
        convo: list[tuple[str, str]] = []
        for message in messages:
            text = normalize(message.content)
            if isinstance(message, SystemMessage):
                if text:
                    system_parts.append(text)
            elif isinstance(message, HumanMessage):
                convo.append(("User", text))
            elif isinstance(message, AIMessage):
                convo.append(("Assistant", text))
            else:
                role = getattr(message, "type", "user") or "user"
                convo.append((role.capitalize(), text))

        system_prompt = "\n\n".join(p for p in system_parts if p)
        if len(convo) == 1 and convo[0][0] == "User":
            prompt = convo[0][1]
        else:
            prompt = "\n\n".join(f"{role}: {text}" for role, text in convo)
        return system_prompt, prompt

    def _build_options(self, system_prompt: str | None) -> ClaudeAgentOptions:
        """Construct ``ClaudeAgentOptions``, gated against SDK field drift."""
        candidate: dict[str, Any] = {
            "model": self.model,
            "system_prompt": system_prompt or None,
            "max_turns": 1,
            # tools=[] → SDK emits `--tools ""` which disables ALL built-in CLI
            # tools. Without it the SDK sends `--tools default`, the model may
            # emit a tool_use, and max_turns=1 then aborts with
            # "Reached maximum number of turns (1)". Pure text generation only.
            "tools": [],
            "permission_mode": "bypassPermissions",
            "env": _get_cli_env(),
            "mcp_servers": {},
            "setting_sources": None,
        }
        cli_path = _resolve_cli_path()
        if cli_path:
            candidate["cli_path"] = cli_path

        accepted = _accepted_option_fields()
        filtered = {k: v for k, v in candidate.items() if k in accepted}
        return ClaudeAgentOptions(**filtered)

    # ── Core query ────────────────────────────────────────────────────
    async def _run_query(self, system_prompt: str, prompt: str) -> str:
        options = self._build_options(system_prompt)
        text_parts: list[str] = []
        result_text: str | None = None

        # One CLI subprocess == one subscription session; charge it before spawning
        # (see the per-run budget in _subscription). Raises AgentBudgetExceeded once
        # the run is spent — the backstop for callers that cannot degrade.
        note_agent_call()

        async with get_concurrency_semaphore():
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    async for message in query(prompt=prompt, options=options):
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    text_parts.append(block.text)
                        elif isinstance(message, ResultMessage):
                            if message.result:
                                result_text = message.result
            except TimeoutError as exc:
                raise RuntimeError(
                    f"[claude_agent] query timed out after {self.timeout_seconds}s"
                ) from exc
            except Exception as exc:
                # The CLI can exit non-zero after delivering all content
                # (process-cleanup race). Tolerate it when we have output.
                if text_parts or result_text:
                    logger.warning("[claude_agent] post-content error ignored: %s", exc)
                else:
                    raise RuntimeError(f"[claude_agent] query failed: {exc}") from exc

        final = result_text if result_text else "".join(text_parts)
        if not final or not final.strip():
            raise RuntimeError(
                "[claude_agent] empty response (possible rate limit or auth failure)"
            )
        # A REFUSAL BANNER IS NOT AN ANSWER. The CLI reports a spent quota or a broken
        # login by printing one line and exiting non-zero — so `result_text` is non-empty,
        # the exception above is swallowed as a "post-content" race, and the banner is
        # returned as if the model had said it. Measured 2026-08-03: every LLM call in a
        # deep_research run returned "You've hit your weekly limit · resets Aug 4, 9pm
        # (UTC)"; the query classifier took it as a CATEGORY, the sub-query generator
        # parsed it into ZERO queries, and the run finished in 16s reporting success with
        # a 0-char, 0-source report. Nothing anywhere said the LLM had never answered.
        # Bounded by length as well as by pattern: a banner is one line, so a real answer
        # that happens to discuss rate limits cannot trip this.
        stripped = final.strip()
        if len(stripped) < 300 and _CLI_REFUSAL_RE.search(stripped):
            raise RuntimeError(
                f"[claude_agent] the CLI refused the request instead of answering: "
                f"{stripped[:200]}"
            )
        return final

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, prompt = self._to_sdk_inputs(messages)
        text = await self._run_query(system_prompt, prompt)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        system_prompt, prompt = self._to_sdk_inputs(messages)
        options = self._build_options(system_prompt)
        got_any = False

        note_agent_call()  # same budget as _run_query — a stream is one session too

        async with get_concurrency_semaphore():
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    async for message in query(prompt=prompt, options=options):
                        if not isinstance(message, AssistantMessage):
                            continue
                        for block in message.content:
                            if isinstance(block, TextBlock) and block.text:
                                got_any = True
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=block.text)
                                )
                                if run_manager is not None:
                                    await run_manager.on_llm_new_token(block.text, chunk=chunk)
                                yield chunk
            except TimeoutError as exc:
                raise RuntimeError(
                    f"[claude_agent] stream timed out after {self.timeout_seconds}s"
                ) from exc
            except Exception as exc:
                if got_any:
                    logger.warning("[claude_agent] post-content stream error ignored: %s", exc)
                else:
                    raise RuntimeError(f"[claude_agent] stream failed: {exc}") from exc
