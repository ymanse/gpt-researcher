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
import threading
import time
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


# ── Per-run agent-call budget ─────────────────────────────────────────
#
# Every generation spawns its own `claude` CLI subprocess (chat_model._run_query),
# and each subprocess registers as a SEPARATE Claude Code session against the
# subscription. Measured 2026-08-05: the container's /root/.claude/projects/-app
# gained 42 session files during the 10 minutes of one research round (11:13-11:23,
# peaking at 9 in a single minute). At tree scale — max_nodes 20, each node running
# its own GPTResearcher for 8-15 calls — a single deep_tree_research is a few
# hundred sessions, which is what shows up on the account as "agents".
#
# Nothing above bounds that count. The semaphore caps how many run AT ONCE (and only
# per event loop: smart_retriever's ThreadPoolExecutor + asyncio.run gives each worker
# thread its own allowance), while the tree's token/credit budgets are denominated in
# quantities a CLI spawn does not move. So the spawn count gets a budget of its own.
#
# Scope is the process, not a task/context: the calls to bound come from asyncio tasks
# AND from retriever worker threads, and a ContextVar does not survive the
# ThreadPoolExecutor hop. A plain lock-guarded counter sees all of them.
#
# fail-open by default: the budget applies only after an explicit begin_agent_run(),
# which the MCP tools call on entry. Library callers (the search-quality harness
# imports gpt_researcher directly) keep the old unbounded behaviour and just get a
# counter they can read.

_CALL_LOCK = threading.Lock()
_calls_spent = 0
_call_limit = 0  # 0 → unbounded; set by begin_agent_run()
# THIS run's own allowance, not the pooled ceiling. The synthesis reserve is a
# fraction of it — see agent_synthesis_reserve for why the pooled ceiling is the
# wrong denominator.
_run_allowance = 0

DEFAULT_MAX_CALLS_PER_RUN = 100


class AgentBudgetExceeded(RuntimeError):
    """A run tried to spawn more CLI sessions than its budget allows."""


DEFAULT_SESSION_RETENTION_DAYS = 2


def prune_cli_sessions(retention_days: float | None = None) -> int:
    """Delete `claude` CLI session transcripts older than the retention window.

    Every LLM call here spawns a CLI subprocess, and each one leaves a transcript
    under ~/.claude/projects/<slug>/*.jsonl that NOTHING ever removes. Measured
    2026-08-19 on the MCP container: 3013 files / 148 MB over 14 days, and — because
    the CLI authenticates with the operator's own CLAUDE_CODE_OAUTH_TOKEN — every one
    of them also shows up in that account's agent list, which is what made a single
    100-call run read as "100 agents appeared".

    mtime-based on purpose: a transcript still being appended to by a live session is
    younger than the window and survives. Best-effort — a run must never fail because
    housekeeping could not delete a file.
    """
    if retention_days is None:
        try:
            retention_days = float(os.environ.get(
                "CLAUDE_AGENT_SESSION_RETENTION_DAYS",
                str(DEFAULT_SESSION_RETENTION_DAYS)))
        except ValueError:
            retention_days = DEFAULT_SESSION_RETENTION_DAYS
    if retention_days <= 0:  # explicit opt-out
        return 0

    root = Path(os.path.expanduser("~")) / ".claude" / "projects"
    if not root.is_dir():
        return 0

    cutoff = time.time() - retention_days * 86400
    removed = 0
    for transcript in root.glob("*/*.jsonl"):
        try:
            if transcript.stat().st_mtime < cutoff:
                transcript.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("Pruned %s claude CLI session transcript(s) older than %sd",
                    removed, retention_days)
    return removed


def begin_agent_run(limit: int | None = None) -> int:
    """Grant one run its allowance of CLI sessions. Returns the limit.

    ``limit=None`` reads ``CLAUDE_AGENT_MAX_CALLS_PER_RUN`` (default 100).
    A limit of 0 or less disarms the budget (unbounded).

    The allowance is ADDED to the ceiling, never reset onto it. Resetting was the
    bug: the counter is process-wide on purpose (a ContextVar cannot follow the
    retriever's ThreadPoolExecutor hop -- see above), so a second concurrent run
    arriving here erased what the first had already spent. Measured 2026-08-06:
    four MCP streams armed two minutes apart (13:51/13:53/13:55/13:57) ended up
    sharing ONE 100-session allowance, burned it in 12 minutes, and failed together.

    Ceiling: allowances POOL rather than isolate. A runaway run can still eat a
    concurrent one's share -- it just can no longer delete the record of its own
    spend, so N runs get N*limit instead of collapsing to one. True per-run
    isolation needs a run id threaded through the thread hop, which is a much
    bigger change than this failure justifies.
    """
    global _call_limit, _run_allowance
    if limit is None:
        try:
            limit = int(os.environ.get(
                "CLAUDE_AGENT_MAX_CALLS_PER_RUN", str(DEFAULT_MAX_CALLS_PER_RUN)))
        except ValueError:
            limit = DEFAULT_MAX_CALLS_PER_RUN
    limit = max(0, limit)
    with _CALL_LOCK:
        _call_limit = (_calls_spent + limit) if limit else 0
        _run_allowance = limit
        ceiling, spent = _call_limit, _calls_spent
    logger.info("Claude Agent call budget armed: +%s (ceiling=%s, already spent=%s)",
                limit or "unbounded", ceiling or "unbounded", spent)
    # housekeeping at the one point every run passes through — see prune_cli_sessions
    prune_cli_sessions()
    return limit


def note_agent_call() -> int:
    """Charge one CLI spawn to the budget. Returns the new spend.

    Raises ``AgentBudgetExceeded`` when the run is already at its limit. This is the
    fail-closed backstop — callers that can degrade gracefully (the tree's batch loop)
    should test ``agent_budget_exhausted()`` first and stop expanding instead, so the
    run still synthesizes a report from what it already gathered.
    """
    global _calls_spent
    with _CALL_LOCK:
        if _call_limit and _calls_spent >= _call_limit:
            raise AgentBudgetExceeded(
                f"[claude_agent] CLI-session budget is spent "
                f"({_calls_spent}/{_call_limit} across the runs armed so far); "
                f"raise CLAUDE_AGENT_MAX_CALLS_PER_RUN or narrow the research scope"
            )
        _calls_spent += 1
        return _calls_spent


def agent_calls_spent() -> int:
    """CLI sessions spawned in this process. Read against ``agent_budget_limit()``,
    which rises by one allowance per run: the pair is what has meaning, not either
    number alone (the counter no longer restarts at each begin_agent_run)."""
    with _CALL_LOCK:
        return _calls_spent


def agent_budget_limit() -> int:
    """Current per-run limit; 0 when unbounded."""
    with _CALL_LOCK:
        return _call_limit


def agent_budget_exhausted(reserve: int = 0) -> bool:
    """True when the next call would raise. Always False when unbounded.

    ``reserve`` holds back that many calls: a caller that still has work to do AFTER
    it stops expanding passes the size of that tail, so the tail is not left broke.
    """
    with _CALL_LOCK:
        return bool(_call_limit) and _calls_spent >= max(0, _call_limit - reserve)


def agent_synthesis_reserve() -> int:
    """Calls to keep back from expansion for the synthesis that follows it.

    Measured 2026-08-05 with a limit of 20: expansion spent all 20, and the roll-up's
    claim-equivalence judge then failed 4 calls in a row and the merge fell back to
    word-overlap — a report that is both shallow AND badly merged, when only the first
    was asked for. Proportional so it scales with the limit rather than starving small
    test budgets: 15%, floor 5.

    Denominated in THIS run's allowance, not the pooled ceiling. The ceiling was the
    bug: allowances pool (see begin_agent_run), so a fourth 100-call run in the same
    process saw a ceiling of 595 and held back 89 — against a spend already at 495,
    that left ELEVEN calls for expansion. Measured 2026-08-16: four sequential tree
    runs got 47/35/21/11 calls of expansion headroom and researched 8/8/7/4 nodes,
    the last stopping on a reserve meant to protect a synthesis that needs ~15.
    """
    with _CALL_LOCK:
        allowance = _run_allowance
    if not allowance:
        return 0
    return max(5, allowance * 15 // 100)
