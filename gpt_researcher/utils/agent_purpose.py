"""Which call site a `claude` CLI session was spawned for.

Every LLM generation on the ``claude_agent`` provider spawns a CLI subprocess that
registers as its own session on the subscription, and a per-run budget bounds how many
a run may spawn (see ``llm_provider/claude_agent/_subscription.py``). The budget could
say a run was expensive; it could never say WHICH of the seven call sites spent it, so
the slimming work — classify once per run, choose the agent once per run, bypass the
planner — had nothing to measure itself against, before or after.

This module is the tag. It lives here, and not next to the counter, for one reason:
importing ``_subscription`` reaches ``claude_agent/__init__.py``, which imports
``chat_model``, which imports the Agent SDK. The sites that need to be tagged are the
retriever's classifier, the planner and the agent chooser — code that also runs on the
OpenRouter rollback path, where the SDK need not be installed at all. ``gpt_researcher.utils``
pulls in nothing, so tagging costs those sites nothing. ``_subscription`` re-exports
``SITES`` and ``agent_purpose`` so callers that already depend on it keep one import.

A ContextVar rather than a thread-local: the tag has to ride an ``await`` chain (the tree's
answer and expansion calls) AND a thread hop (``SmartRetriever`` is synchronous and hops off
the running loop to drive its coroutine). ContextVars follow both — with one caveat that
cost a measurement to find: ``ThreadPoolExecutor.submit`` does NOT copy the calling
context, so the submit site has to carry it across by hand
(``smart_retriever._run_coro_blocking``). Without that, every production ``classify``
charge lands under ``untagged`` and the "classify + choose_agent <= 2 per run" gate passes
vacuously by reading zero.

Sibling isolation comes for free: ``asyncio.gather`` copies the context per Task, so two
tree nodes researched concurrently cannot be charged to each other's site.
"""
from __future__ import annotations

import contextlib
import contextvars

UNTAGGED = "untagged"

#: The call sites a run's spend is broken down by. ``untagged`` is not a site — it is
#: what a call that nobody wrapped is charged to, and a production run reporting a large
#: ``untagged`` share means a site was missed, not that the work was anonymous.
SITES = (
    "choose_agent",   # actions/agent_creator.choose_agent
    "classify",       # retrievers/smart/smart_retriever._classify_query
    "plan",           # actions/query_processing.plan_research_outline
    "answer",         # skills/tree_research.research_node
    "children",       # skills/tree_research.generate_child_questions
    "merge",          # skills/tree_research, the claim-equivalence judge
    "verify",         # gptr-mcp/verification.verify_research
    UNTAGGED,
)

_purpose: contextvars.ContextVar[str] = contextvars.ContextVar(
    "gptr_agent_purpose", default=UNTAGGED)


@contextlib.contextmanager
def agent_purpose(site: str):
    """Charge every CLI session spawned in this block to ``site``.

    Unknown names are recorded verbatim rather than rejected: a research run must not die
    because a telemetry label was misspelled, and a stray key in the breakdown is a louder
    signal than a silent remap would be.
    """
    token = _purpose.set(str(site) if site else UNTAGGED)
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> str:
    """The site in effect on this context, or ``untagged`` outside every block."""
    return _purpose.get()
