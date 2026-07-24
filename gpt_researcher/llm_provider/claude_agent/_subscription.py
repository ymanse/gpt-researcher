"""Subscription-auth helpers for the Claude Agent SDK provider.

Trimmed Linux-only port of the auto-rpg ``adapters/_shared.py`` helpers. The
Claude Agent SDK wraps the ``claude`` CLI as a subprocess; this module builds a
clean subprocess environment that authenticates with the user's Claude
Pro/Max **subscription** (via ``CLAUDE_CODE_OAUTH_TOKEN``) rather than the
pay-per-token Anthropic API.

Billing correctness (critical): ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``
are scrubbed from the subprocess env dict so the CLI never falls back to the
metered API. Only the subprocess dict is altered — the parent ``os.environ`` is
left untouched so OpenAI embeddings keep working.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Process-level env neutralization (run once at import) ─────────────
# The SDK's version check runs subprocess.run() with bare os.environ.
# Setting these at module load ensures that subprocess inherits sane values.
os.environ.pop("CLAUDECODE", None)
os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"


# ── Model resolution ──────────────────────────────────────────────────

# Canonical short-alias → full model ID mapping.
_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
}


def _resolve_model_id(model: str) -> str:
    """Resolve a short alias to a full Anthropic model identifier.

    Maps ``"sonnet"`` / ``"opus"`` / ``"haiku"`` to the latest model IDs.
    When the matching ``ANTHROPIC_DEFAULT_<ALIAS>_MODEL`` env var is set, that
    override wins. Unrecognised strings (already-qualified IDs) pass through
    unchanged.
    """
    model_lower = model.lower()

    if model_lower == "sonnet":
        env_model = os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL")
        if env_model:
            return env_model
    elif model_lower == "opus":
        env_model = os.getenv("ANTHROPIC_DEFAULT_OPUS_MODEL")
        if env_model:
            return env_model
    elif model_lower == "haiku":
        env_model = os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL")
        if env_model:
            return env_model

    return _MODEL_MAP.get(model_lower, model)


# ── CLI path resolution (Unix only) ───────────────────────────────────


def _resolve_cli_path() -> str | None:
    """Find the Claude CLI binary path explicitly (Linux container).

    The MCP server subprocess may have a different PATH than the install,
    so we resolve the path here and pass it via ``ClaudeAgentOptions(cli_path=...)``.
    Returns ``None`` to let the SDK auto-detect when nothing is found.
    """
    # 1. Env override
    env_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_path:
        resolved = Path(env_path)
        if resolved.is_file():
            logger.debug("CLI path from CLAUDE_CLI_PATH env: %s", resolved)
            return str(resolved)
        logger.debug("CLAUDE_CLI_PATH set but not found: %s", env_path)

    # 2. shutil.which (works in the current process env)
    which_path = shutil.which("claude")
    if which_path:
        logger.debug("CLI path from shutil.which: %s", which_path)
        return which_path

    # 3. Known Unix install locations
    candidates = [
        Path("/usr/local/bin/claude"),
        Path.home() / ".local" / "bin" / "claude",
    ]
    for p in candidates:
        if p.is_file():
            logger.debug("CLI path from known location: %s", p)
            return str(p)

    logger.debug("CLI path resolution failed — letting SDK auto-detect")
    return None


# ── CLI environment ───────────────────────────────────────────────────


def _get_cli_env() -> dict[str, str]:
    """Build a clean subprocess env for the Claude CLI.

    - Removes ``CLAUDECODE`` so the CLI does not detect a nested instance.
    - **Scrubs ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``** so billing
      cannot revert to the metered Anthropic API. Only the returned dict is
      modified — the parent ``os.environ`` (and thus OpenAI embeddings) is
      untouched.
    - Sets SDK/encoding flags and clears ``PYTHONPATH`` to avoid pollution.
    - Forwards ``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_AUTH_TOKEN`` — the
      subscription auth — which are carried over by ``os.environ.copy()``.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Billing correctness: never let the CLI fall back to the metered API.
    if env.pop("ANTHROPIC_API_KEY", None):
        logger.debug("Scrubbed ANTHROPIC_API_KEY from CLI subprocess env (subscription only)")
    env.pop("ANTHROPIC_BASE_URL", None)

    env["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = ""

    # The SDK runs the CLI with permission_mode="bypassPermissions"
    # (→ --dangerously-skip-permissions), which the CLI refuses under root.
    # This container runs as root, so mark it a sandbox — the same mechanism
    # Anthropic's official devcontainer uses to run the CLI as root.
    env.setdefault("IS_SANDBOX", "1")

    if env.get("CLAUDE_CODE_OAUTH_TOKEN") or env.get("ANTHROPIC_AUTH_TOKEN"):
        logger.debug("Subscription auth token present in CLI subprocess env")
    else:
        logger.warning(
            "No CLAUDE_CODE_OAUTH_TOKEN/ANTHROPIC_AUTH_TOKEN in env — CLI will "
            "rely on its own stored OAuth (may fail in a fresh container)"
        )
    return env


# ── Concurrency control ───────────────────────────────────────────────

# Per-event-loop semaphores: smart_retriever runs coros via asyncio.run() in worker
# threads, so a single process-wide semaphore gets bound to a transient loop and every
# later call on the main loop dies with "bound to a different event loop".
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def get_concurrency_semaphore() -> asyncio.Semaphore:
    """Lazily build a per-event-loop semaphore bounding concurrent CLI calls.

    Bound is ``CLAUDE_AGENT_MAX_CONCURRENCY`` (default 2) to stay within
    subscription rate limits. One semaphore per running loop; dead loops drop
    out of the WeakKeyDictionary automatically.
    """
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        try:
            limit = int(os.environ.get("CLAUDE_AGENT_MAX_CONCURRENCY", "2"))
        except ValueError:
            limit = 2
        limit = max(1, limit)
        sem = asyncio.Semaphore(limit)
        _semaphores[loop] = sem
        logger.debug("Claude Agent SDK concurrency semaphore initialized: limit=%d", limit)
    return sem
