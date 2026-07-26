"""Utility functions for GPT Researcher retrievers.

This module provides helper functions and constants used by the
various search retriever implementations.
"""

import importlib.util
import logging
import os
import re

logger = logging.getLogger(__name__)


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
