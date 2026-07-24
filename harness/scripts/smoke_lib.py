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


def smoke_preamble(stage: int) -> dict:
    """recreate + health; returns the common evidence header (fail-closed values on error)."""
    ok = recreate()
    health = wait_health() if ok else 0
    return {"stage": stage, "recreated": bool(ok), "health": health,
            "query": hconf.SMOKE_QUERIES[stage]}


def finish(stage: int, ev: dict) -> None:
    hconf.write_json(hconf.EVID / f"stage{stage}_smoke.json", ev)
    print(json.dumps(ev, indent=2))
