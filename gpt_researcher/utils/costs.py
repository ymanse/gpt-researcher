"""Cost estimation utilities for LLM API usage.

This module provides functions to estimate the cost of LLM API calls
based on token counts. Cost estimates are based on OpenAI pricing
and may vary for other model providers.
"""

import tiktoken

# Per OpenAI Pricing Page: https://openai.com/api/pricing/
ENCODING_MODEL = "o200k_base"
INPUT_COST_PER_TOKEN = 0.000005
OUTPUT_COST_PER_TOKEN = 0.000015
IMAGE_INFERENCE_COST = 0.003825
EMBEDDING_COST = 0.02 / 1000000  # Assumes new ada-3-small


def _coerce_to_text(value) -> str:
    """Coerce arbitrary LLM input/output to a tiktoken-encodable str.

    Gemini AFC (Automatic Function Calling) responses and some provider SDKs
    return content as `list[ContentPart]` or `list[str]` instead of `str`.
    `tiktoken.Encoding.encode` strictly requires `str`, so we flatten lists
    to text here. Also handles None / non-str primitives.
    """
    if value is None:
        return ""
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
                # Common shapes: {"text": "..."} / {"content": "..."} / {"role": ..., "content": ...}
                for key in ("text", "content", "value"):
                    if key in item and isinstance(item[key], str):
                        parts.append(item[key])
                        break
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def estimate_llm_cost(input_content, output_content) -> float:
    """Estimate the cost of an LLM API call based on input and output content.

    Cost estimation is based on OpenAI pricing and may vary for other models.

    Args:
        input_content: The input text sent to the LLM. Normally `str`; lists/dicts
            (Gemini AFC, multi-LLM review responses) are coerced via `_coerce_to_text`.
        output_content: The output text received from the LLM. Same coercion as input.

    Returns:
        The estimated cost in USD.
    """
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    # B-tier permanent patch: coerce list/dict/None to str to avoid
    # tiktoken `TypeError: argument 'text': 'list' object is not an instance of 'str'`
    # See https://github.com/assafelovic/gpt-researcher/issues/1022
    input_text = _coerce_to_text(input_content)
    output_text = _coerce_to_text(output_content)
    input_tokens = encoding.encode(input_text, disallowed_special=())
    output_tokens = encoding.encode(output_text, disallowed_special=())
    input_costs = len(input_tokens) * INPUT_COST_PER_TOKEN
    output_costs = len(output_tokens) * OUTPUT_COST_PER_TOKEN
    return input_costs + output_costs


def estimate_embedding_cost(model: str, docs: list) -> float:
    """Estimate the cost of embedding documents.

    Args:
        model: The embedding model name.
        docs: List of documents to embed.

    Returns:
        The estimated embedding cost in USD.
    """
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(str(doc), disallowed_special=())) for doc in docs)
    return total_tokens * EMBEDDING_COST

