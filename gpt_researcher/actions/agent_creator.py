"""Agent creation and selection utilities for GPT Researcher.

This module provides functions to automatically select and configure
the appropriate research agent based on the query type.
"""

import json
import logging
import re

import json_repair

from ..prompts import PromptFamily
from ..utils.agent_purpose import agent_purpose
from ..utils.llm import create_chat_completion

logger = logging.getLogger(__name__)

async def choose_agent(
    query,
    cfg,
    parent_query=None,
    cost_callback: callable = None,
    headers=None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
):
    """
    Chooses the agent automatically
    Args:
        parent_query: In some cases the research is conducted on a subtopic from the main query.
            The parent query allows the agent to know the main context for better reasoning.
        query: original query
        cfg: Config
        cost_callback: callback for calculating llm costs
        prompt_family: Family of prompts

    Returns:
        agent: Agent name
        agent_role_prompt: Agent role prompt
    """
    query = f"{parent_query} - {query}" if parent_query else f"{query}"
    response = None  # Initialize response to ensure it's defined

    try:
        # One CLI session per node today, for a role prompt the tree could resolve once
        # for the whole run — see P1.2. Tagged so the saving is measurable either way.
        with agent_purpose("choose_agent"):
            response = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": f"{prompt_family.auto_agent_instructions()}"},
                    {"role": "user", "content": f"task: {query}"},
                ],
                temperature=0.15,
                llm_provider=cfg.smart_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs
            )

        # B-tier permanent patch: some LLM providers (Gemini AFC, multi-LLM review)
        # return `list[ContentPart]` or `list[str]` instead of `str`.
        # json.loads/json_repair.loads require str/bytes — coerce here.
        response = _coerce_response_to_text(response)

        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]

    except Exception as e:
        return await handle_json_error(response)


def _coerce_response_to_text(value) -> str | None:
    """Coerce arbitrary LLM response to str for downstream json/regex parsing.

    Gemini AFC (Automatic Function Calling) and some provider SDKs return
    content as `list[ContentPart]`/`list[str]` instead of `str`. This helper
    normalizes those shapes to a single concatenated string so that
    `json.loads`, `json_repair.loads`, regex extraction, and slicing all work.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    inner = item.get(key)
                    if isinstance(inner, str):
                        parts.append(inner)
                        break
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


async def handle_json_error(response):
    """Handle JSON parsing errors from LLM responses.

    Attempts to recover agent information from malformed JSON responses
    using json_repair and regex extraction as fallbacks.

    Args:
        response: The LLM response that failed initial JSON parsing.
            Normally `str`; lists/dicts (Gemini AFC) are coerced to str here.

    Returns:
        A tuple of (agent_name, agent_role_prompt). Returns default agent
        if all parsing attempts fail.
    """
    # B-tier permanent patch: ensure response is str before all downstream parsers
    response = _coerce_response_to_text(response)

    try:
        agent_dict = json_repair.loads(response)
        # json_repair may return a string instead of a dict for malformed input;
        # guard `.get` access.
        if isinstance(agent_dict, dict) and agent_dict.get("server") and agent_dict.get("agent_role_prompt"):
            return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(
            f"Failed to parse agent JSON with json_repair: {error_type}: {error_msg}",
            exc_info=True
        )
        if response:
            logger.debug(f"LLM response that failed to parse: {response[:500]}...")

    json_string = extract_json_with_regex(response)
    if json_string:
        try:
            json_data = json.loads(json_string)
            return json_data["server"], json_data["agent_role_prompt"]
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to decode JSON from regex extraction: {str(e)}",
                exc_info=True
            )

    logger.info("No valid JSON found in LLM response. Falling back to default agent.")
    return "Default Agent", (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )


def extract_json_with_regex(response: str | None) -> str | None:
    """Extract JSON object from a string using regex.

    Attempts to find the first JSON object pattern in the response string.

    Args:
        response: The string to search for JSON content.

    Returns:
        The extracted JSON string if found, None otherwise.
    """
    if not response:
        return None
    json_match = re.search(r"{.*?}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None
