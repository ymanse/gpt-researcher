"""Shared live-smoke plumbing: container recreate, health wait, MCP streamable-http calls,
docker-log harvesting of TIERA_EVIDENCE lines.

Evidence JSONs written here are the ONLY thing the smoke gates read — every field derives
from the container's own behavior (health endpoint, MCP response, docker logs), never from
an agent's claim. Re-running a smoke script is its law-10 regenerator.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import re
import subprocess
import time
import urllib.request
from datetime import timedelta

import hconf

TIERA_RE = re.compile(r"TIERA_EVIDENCE stage=(\d+) (.*)")


def recreate() -> bool:
    r = subprocess.run(
        ["docker", "compose", "-f", hconf.COMPOSE_FILE, "up", "-d",
         "--force-recreate", hconf.COMPOSE_SERVICE],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def wait_health(timeout_s: int = 240) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(hconf.HEALTH_URL, timeout=5) as resp:
                if resp.status == 200 and b"healthy" in resp.read():
                    return 200
        except OSError:
            pass
        time.sleep(3)
    return 0


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def docker_logs_since(since_iso: str) -> str:
    r = subprocess.run(
        ["docker", "logs", hconf.CONTAINER, "--since", since_iso],
        capture_output=True, text=True, errors="replace",
    )
    return (r.stdout or "") + "\n" + (r.stderr or "")


def tiera_lines(logs: str, stage: int) -> list[dict[str, int]]:
    """Parse `TIERA_EVIDENCE stage=N k=v ...` lines emitted by the implementation."""
    out = []
    for m in TIERA_RE.finditer(logs):
        if int(m.group(1)) != stage:
            continue
        kv = {}
        for pair in m.group(2).split():
            if "=" in pair:
                k, _, v = pair.partition("=")
                try:
                    kv[k] = int(v)
                except ValueError:
                    kv[k] = v
        out.append(kv)
    return out


async def _call(tool: str, arguments: dict, timeout_s: int):
    from mcp import ClientSession
    # ponytail: deliberately the deprecated entry point — the new streamable_http_client
    # dropped timeout/sse_read_timeout (needs a hand-built httpx client with MCP headers);
    # this one still takes them directly, which deep_research's hour-long calls require.
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        hconf.MCP_URL,
        timeout=timedelta(seconds=120),
        sse_read_timeout=timedelta(seconds=timeout_s),
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                tool, arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout_s),
            )
            if getattr(res, "structuredContent", None):
                return res.structuredContent
            for c in res.content or []:
                text = getattr(c, "text", None)
                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"text": text}
            return {}


def mcp_call(tool: str, arguments: dict, timeout_s: int = 3600) -> dict:
    return asyncio.run(_call(tool, arguments, timeout_s))


def exc_summary(e: BaseException) -> str:
    """Flatten ExceptionGroup leaves so evidence records the real error, not
    'unhandled errors in a TaskGroup'."""
    leaves: list[str] = []

    def walk(x: BaseException) -> None:
        if isinstance(x, BaseExceptionGroup):
            for sub in x.exceptions:
                walk(sub)
        else:
            leaves.append(f"{type(x).__name__}: {x}")

    walk(e)
    return " | ".join(leaves)[:400]


def _pid_alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
    return str(pid) in r.stdout


def _acquire_smoke_lock() -> None:
    """One smoke at a time: every smoke force-recreates the SHARED container, so a second
    concurrent run kills the first one's MCP session mid-call (measured: 43 orphaned smoke
    processes recreating over each other produced 'unhandled errors in a TaskGroup').
    Refuse to start while another smoke is alive; stale locks self-clear."""
    import atexit
    import os
    lock = hconf.HARNESS / "no_read" / "smoke.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = 0
        if pid and _pid_alive(pid):
            print(f"SMOKE LOCKED: another smoke run (PID {pid}) is in progress — do NOT start "
                  f"a second one (it would force-recreate the shared container and kill the "
                  f"running MCP call). Wait for it to finish, then read the evidence it writes.")
            raise SystemExit(2)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))


def smoke_preamble(stage: int) -> dict:
    """recreate + health; returns the common evidence header (fail-closed values on error)."""
    _acquire_smoke_lock()
    ok = recreate()
    health = wait_health() if ok else 0
    return {"stage": stage, "recreated": bool(ok), "health": health,
            "query": hconf.SMOKE_QUERIES[stage]}


def finish(stage: int, ev: dict) -> None:
    hconf.write_json(hconf.EVID / f"stage{stage}_smoke.json", ev)
    print(json.dumps(ev, indent=2))
