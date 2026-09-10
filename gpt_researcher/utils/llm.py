"""LLM utilities for GPT Researcher.

This module provides utility functions for interacting with various
LLM providers through a unified interface.
"""
from __future__ import annotations

import logging
import os
from typing import Any
import asyncio

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import (
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
)

from ..prompts import PromptFamily
from .costs import estimate_llm_cost
from .validators import Subtopics


# Raised by ChatClaudeAgent._run_query (gpt_researcher/llm_provider/claude_agent/
# chat_model.py) only after it has already classified the CLI's stdout as a refusal
# banner (spent weekly quota / invalid key / not logged in), never an actual model
# reply. Matched by prefix instead of re-deriving the classification: chat_model.py
# owns the "is this a refusal" judgement, this only recognizes its verdict.
_CLI_REFUSAL_PREFIX = "[claude_agent] the CLI refused the request instead of answering:"

# Same shape as retrievers.utils.is_retryable_error: whitelist what is NOT retryable,
# default everything else to retryable so an error with no evidence either way is not
# silently made fatal. 401/403 are HTTP auth/authz semantics — the identical request
# meets the identical rejection, same reasoning the retriever layer already applies to
# 401/403 (see retrievers/utils.py _NON_RETRYABLE_STATUS).
_LLM_NON_RETRYABLE_STATUS = {401, 403}


def is_llm_retryable_error(exc: BaseException) -> bool:
    """True if retrying `exc` from create_chat_completion could plausibly succeed.

    Measured live 2026-08-04 (harness-search/no_read/tmp/probe_node_timing_result.json):
    one `conduct_research()` call spent 395.19s wall for zero usable context. All 50 LLM
    calls failed with the identical CLI refusal (spent weekly quota) across 5
    `create_chat_completion()` invocations, each burning a full 10-attempt exponential
    backoff (55s of asyncio.sleep) on a condition no retry could ever clear — 271 of the
    395s (69%) was sleep between doomed retries, not work.
    """
    if str(exc).startswith(_CLI_REFUSAL_PREFIX):
        # Record it as well as refusing to retry. Classifying a refusal without
        # REMEMBERING it is what let the expansion loops keep going into a provider that
        # had stopped answering, so the next call killed the whole run instead of the run
        # finishing with what it had. Import here, not at module scope: claude_agent
        # drags in the Agent SDK and this module is loaded by every provider.
        try:
            from gpt_researcher.llm_provider.claude_agent._subscription import (
                note_provider_refusal,
            )
        except ImportError:
            pass
        else:
            note_provider_refusal(str(exc)[len(_CLI_REFUSAL_PREFIX):].strip())
        return False
    # Same shape as the refusal above, and it cost the same way: the per-run CLI-session
    # budget is spent, and no amount of backoff refills it. Measured 2026-08-06: four MCP
    # research streams each burned 10 attempts (55s of sleep) per call against an
    # exhausted budget, which is what turned one spent budget into 3 client timeouts and
    # 1 surfaced error. Imported here rather than at module scope: claude_agent drags the
    # Agent SDK in on package import, and this module is loaded by every provider.
    try:
        from gpt_researcher.llm_provider.claude_agent._subscription import (
            AgentBudgetExceeded,
        )
    except ImportError:
        pass
    else:
        if isinstance(exc, AgentBudgetExceeded):
            return False
    # openai/anthropic SDK APIStatusError exposes status_code directly on the exception;
    # requests.HTTPError nests it under .response.status_code; urllib.error.HTTPError
    # uses .code. Check all three, same fallback order as is_retryable_error.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if status is None:
        return True
    return status not in _LLM_NON_RETRYABLE_STATUS


def get_llm(llm_provider: str, **kwargs):
    """Get an LLM provider instance.

    Args:
        llm_provider: The name of the LLM provider (e.g., 'openai', 'anthropic').
        **kwargs: Additional keyword arguments passed to the provider.

    Returns:
        A GenericLLMProvider instance configured for the specified provider.
    """
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        **kwargs
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.4.
        max_tokens (int, optional): The max tokens to use. Defaults to 4000.
        llm_provider (str, optional): The LLM Provider to use.
        stream (bool): Whether to stream the response. Defaults to False.
        webocket (WebSocket): The websocket used in the currect request,
        llm_kwargs (dict[str, Any], optional): Additional LLM keyword arguments. Defaults to None.
        cost_callback: Callback function for updating cost.
        reasoning_effort (str, optional): Reasoning effort for OpenAI's reasoning models. Defaults to 'low'.
        **kwargs: Additional keyword arguments.
    Returns:
        str: The response from the chat completion.
    """
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 32001:
        raise ValueError(
            f"Max tokens cannot be more than 32,000, but got {max_tokens}")

    # Get the provider from supported providers
    provider_kwargs = {'model': model}

    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs['reasoning_effort'] = reasoning_effort

    if model not in NO_SUPPORT_TEMPERATURE_MODELS:
        provider_kwargs['temperature'] = temperature
        provider_kwargs['max_tokens'] = max_tokens
    else:
        provider_kwargs['temperature'] = None
        provider_kwargs['max_tokens'] = None

    if llm_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            provider_kwargs['openai_api_base'] = base_url

    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    # create response
    max_attempts = 1 if (stream and websocket is not None) else 10
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await provider.get_chat_response(
                messages, stream, websocket, **kwargs
            )
        except Exception as exc:
            last_exception = exc
            logging.getLogger(__name__).warning(
                f"LLM request failed (attempt {attempt}/{max_attempts}): {exc}"
            )
            if not is_llm_retryable_error(exc):
                logging.getLogger(__name__).warning(
                    f"LLM error is not retryable, failing fast without a wasted "
                    f"retry (attempt {attempt}/{max_attempts}): {exc}"
                )
                break
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        if not response:
            last_exception = RuntimeError("Empty response from LLM provider")
            logging.getLogger(__name__).warning(
                f"LLM returned empty response (attempt {attempt}/{max_attempts})"
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            break

        if cost_callback:
            llm_costs = estimate_llm_cost(str(messages), response)
            cost_callback(llm_costs)

        return response

    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API") from last_exception


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """
    Construct subtopics based on the given task and data.

    Args:
        task (str): The main task or topic.
        data (str): Additional data for context.
        config: Configuration settings.
        subtopics (list, optional): Existing subtopics. Defaults to [].
        prompt_family (PromptFamily): Family of prompts
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of constructed subtopics.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        provider_kwargs = {'model': config.smart_llm_model}

        if config.llm_kwargs:
            provider_kwargs.update(config.llm_kwargs)

        if config.smart_llm_model in SUPPORT_REASONING_EFFORT_MODELS:
            provider_kwargs['reasoning_effort'] = ReasoningEfforts.High.value
        else:
            provider_kwargs['temperature'] = config.temperature
            provider_kwargs['max_tokens'] = config.smart_token_limit

        provider = get_llm(config.smart_llm_provider, **provider_kwargs)

        model = provider.llm

        chain = prompt | model | parser

        output = await chain.ainvoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        }, **kwargs)

        return output

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        logging.getLogger(__name__).error("Exception in parsing subtopics : \n {e}")
        return subtopics
