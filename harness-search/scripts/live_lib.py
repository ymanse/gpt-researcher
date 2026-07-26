"""Shared live-measure plumbing: container recreate, health wait, MCP streamable-http
calls, per-round run caching. Adapted from the proven tier-a smoke_lib.py.

Evidence JSONs written by measure.py/benchmark.py are the ONLY thing measure gates read —
every field derives from the container's behavior (health, MCP response, tree.json,
score_report.py output), never from an agent's claim. Re-running with the round cache
present re-scores from cache — that is the law-10 regenerator for measure evidence.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import urllib.request
from datetime import timedelta

import code_fp
import hconf


def recreate() -> bool:
    r = subprocess.run(
        ["docker", "compose", "-f", hconf.COMPOSE_FILE, "up", "-d",
         "--force-recreate", hconf.COMPOSE_SERVICE],
        capture_output=True, text=True, timeout=600,
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


async def _call(tool: str, arguments: dict, timeout_s: int):
    from mcp import ClientSession
    # ponytail: deliberately the deprecated entry point — the new streamable_http_client
    # dropped timeout/sse_read_timeout; hour-long tree calls need them directly.
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


def mcp_call(tool: str, arguments: dict, timeout_s: int = 5400) -> dict:
    return asyncio.run(_call(tool, arguments, timeout_s))


def exc_summary(e: BaseException) -> str:
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
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True,
                       errors="replace", timeout=60)
    return str(pid) in (r.stdout or "")


def acquire_live_lock(max_wait_s: int = 7200) -> None:
    """One live measure at a time: every run force-recreates the SHARED container, so a
    concurrent second run kills the first one's MCP session mid-call (measured on tier-a).

    A busy lock WAITS rather than exiting. Bailing looked safer but wasn't: the loop
    starts a fresh session every few minutes, and a session that dies on a busy lock
    burns an iteration for nothing (measured: 7 iterations burned while a real 30-min
    measure was in flight). Waiting is also cheap — once the holder finishes it has
    written this round's cache, so the waiter's own run finishes from cache in seconds.
    Stale locks (dead PID) self-clear."""
    import atexit
    import os
    lock = hconf.HARNESS / "no_read" / "live.lock"
    deadline = time.time() + max_wait_s
    waited = 0
    while lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except (ValueError, OSError):
            pid = 0
        if not pid or not _pid_alive(pid):
            break                                   # stale lock: holder is gone
        if time.time() > deadline:
            print(f"LIVE LOCK held by live PID {pid} for more than {max_wait_s}s — refusing "
                  f"to start a second measure (it would recreate the shared container and "
                  f"kill the running MCP call). Investigate that process, then retry.")
            raise SystemExit(2)
        if waited % 300 == 0:
            print(f"[measure] another live run (PID {pid}) is in progress — waiting for it; "
                  f"its round cache will make this run cheap", flush=True)
        time.sleep(15)
        waited += 15
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))


def run_tree_cached(golden: dict, rnd: int, timeout_s: int = 5400) -> tuple[dict | None, str]:
    """deep_tree_research for one golden query, cached per (round, golden, CODE VERSION).

    Returns ({"tree": <path>, "report": <path>}, "") on success or (None, error).

    The code fingerprint is part of the key, not just (round, id): an interrupted run
    still resumes for free, but the moment the implementation changes the artifact is
    re-run live. Without it, an agent that fixes the implementation re-scores the tree
    the OLD code produced, the score never moves, and the gate becomes unsatisfiable —
    the law-6 broken referee, measured on s2 round 0.
    """
    cdir = hconf.BENCH_RUNS / f"round{rnd}"
    cdir.mkdir(parents=True, exist_ok=True)
    tree_p = cdir / f"{golden['id']}.tree.json"
    rep_p = cdir / f"{golden['id']}.report.md"
    fp_p = cdir / f"{golden['id']}.codefp"
    fp = code_fp.fingerprint()
    if tree_p.exists() and rep_p.exists() and fp_p.exists() \
            and fp_p.read_text(encoding="utf-8").strip() == fp:
        return {"tree": str(tree_p), "report": str(rep_p)}, ""
    if tree_p.exists():
        print(f"[measure] {golden['id']}: implementation changed since the cached run "
              f"(code_fp {fp}) — re-running live", flush=True)
    try:
        res = mcp_call("deep_tree_research", {"query": golden["query"]}, timeout_s)
    except BaseException as e:  # noqa: BLE001 — evidence must record the real leaf error
        return None, exc_summary(e)
    tree_host = res.get("tree_json_path") or res.get("tree_path") or ""
    rep_host = res.get("report_path") or res.get("report_md_path") or ""
    if not tree_host or not rep_host:
        return None, f"mcp result lacked tree/report host paths: {str(res)[:300]}"
    try:
        shutil.copyfile(tree_host, tree_p)
        shutil.copyfile(rep_host, rep_p)
        fp_p.write_text(fp + "\n", encoding="utf-8")
    except OSError as e:
        return None, f"copy failed: {e}"
    return {"tree": str(tree_p), "report": str(rep_p)}, ""


def score(golden_id: str, report: str, tree: str | None, out_rel: str) -> dict | None:
    cmd = [str(hconf.VENV_PY), str(hconf.BENCH / "score_report.py"),
           "--golden", str(hconf.GOLDEN / f"{golden_id}.json"),
           "--report", report, "--out", out_rel,
           "--fetch-cache", str(hconf.FETCH_CACHE)]
    if tree:
        cmd += ["--tree", tree]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hconf.HARNESS),
                       timeout=1800)
    p = hconf.HARNESS / out_rel
    if r.returncode != 0 or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
