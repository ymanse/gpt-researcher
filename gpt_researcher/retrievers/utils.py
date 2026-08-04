"""Utility functions for GPT Researcher retrievers.

This module provides helper functions and constants used by the
various search retriever implementations.
"""

import importlib.util
import logging
import os
import re

logger = logging.getLogger(__name__)


# Observed in a live run (harness-search): GithubSearch logged HTTP 422 six times —
# once per sub-query, never recovering, because the request itself is malformed and a
# retry resends the identical request into the identical rejection. 400/401/403/404
# are the same shape (bad/unauthorized/absent request, not a transient condition).
# Everything NOT in this set — 429/502/503/504 (also observed live, worth retrying)
# and any status-less exception (connection refused, DNS failure, read timeout — no
# response was ever received) — defaults to retryable, preserving today's behavior
# for codes with no evidence either way (e.g. Tavily's non-standard 432 rate-limit
# signal) instead of guessing a wider ban.
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


def is_retryable_error(exc: BaseException) -> bool:
    """True if retrying `exc` could plausibly succeed.

    Reads the HTTP status off `requests.HTTPError` (`.response.status_code`) or
    `urllib.error.HTTPError` (`.code`). No status found means no response was ever
    received (network-level failure) — transient by definition, so retryable.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if status is None:
        return True
    return status not in _NON_RETRYABLE_STATUS


def normalize_wikipedia_lang(lang) -> str:
    """Normalize a language or ddgs region code to a valid wikipedia language code.

    ddgs regions come as "country-lang" ("us-en", "wt-wt"); wikipedia only needs
    the lang segment. "wt" is ddgs's "worldwide" placeholder, not a language —
    it builds https://wt.wikipedia.org, which does not resolve.
    """
    if not lang:
        return "en"
    lang = lang.lower().strip().split("-")[-1]
    if lang == "wt" or not re.fullmatch(r"[a-z]{2,3}", lang):
        return "en"
    return lang

async def stream_output(log_type, step, content, websocket=None, with_data=False, data=None):
    """
    Stream output to the client.
    
    Args:
        log_type (str): The type of log
        step (str): The step being performed
        content (str): The content to stream
        websocket: The websocket to stream to
        with_data (bool): Whether to include data
        data: Additional data to include
    """
    if websocket:
        try:
            if with_data:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content,
                    "data": data
                })
            else:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error streaming output: {e}")

def check_pkg(pkg: str) -> None:
    """
    Checks if a package is installed and raises an error if not.
    
    Args:
        pkg (str): The package name
    
    Raises:
        ImportError: If the package is not installed
    """
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        raise ImportError(
            f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )

# Valid retrievers for fallback
VALID_RETRIEVERS = [
    "tavily",
    "custom",
    "duckduckgo",
    "searchapi",
    "serper",
    "serpapi",
    "google",
    "searx",
    "bing",
    "arxiv",
    "semantic_scholar",
    "pubmed_central",
    "exa",
    "mcp",
    "xquik",
    "smart",
    "mock"
]

def get_all_retriever_names():
    """
    Get all available retriever names
    :return: List of all available retriever names
    :rtype: list
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get all items in the current directory
        all_items = os.listdir(current_dir)
        
        # Filter out only the directories, excluding __pycache__
        retrievers = [
            item for item in all_items 
            if os.path.isdir(os.path.join(current_dir, item)) and not item.startswith('__')
        ]
        
        return retrievers
    except Exception as e:
        logger.error(f"Error getting retrievers: {e}")
        return VALID_RETRIEVERS
