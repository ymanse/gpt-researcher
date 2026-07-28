"""Shared constants + helpers for the search-quality harness scripts.

Every evidence file the gates read is produced by these scripts from a real tool's own
output (pytest junit xml, docker, git, score_report.py) — never hand-authored (law 10).
All emitters are idempotent: same inputs, same bytes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import xml.etree.ElementTree as ET

HARNESS = pathlib.Path(__file__).resolve().parents[1]
REPO = HARNESS.parent
MCP_REPO = pathlib.Path("D:/dev_ext/gptr-mcp")
VENV_PY = REPO / "venv" / "Scripts" / "python.exe"
EVID = HARNESS / "no_read" / "evidence"
TESTS = REPO / "tests" / "search_quality"
PYTEST_INI = HARNESS / "pytest.ini"
BENCH = HARNESS / "bench"
GOLDEN = BENCH / "golden"
BASELINE = BENCH / "baseline_firecrawl.json"
MANIFEST = BENCH / "manifest.sha256"
BENCH_RUNS = HARNESS / "no_read" / "bench_runs"
FETCH_CACHE = HARNESS / "no_read" / "fetch_cache"
STORE = HARNESS / ".gralph" / "search-quality" / "store.json"
COMPOSE_FILE = "D:/docker/gptr-mcp/docker-compose.yml"
COMPOSE_SERVICE = "gptr-mcp"
CONTAINER = "gptr-mcp-server"
MCP_URL = "http://127.0.0.1:8123/mcp"
HEALTH_URL = "http://127.0.0.1:8123/health"
OUTPUTS_HOST = MCP_REPO / "outputs"
BRANCH = "feature/search-quality"
TAG = "[sq]"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_get(key: str, default: int | str = 0, instance: str = "search-quality"):
    path = HARNESS / ".gralph" / instance / "store.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key, default)
    except (OSError, json.JSONDecodeError):
        return default


def bench_round() -> int:
    try:
        return int(store_get("bench_round", 0))
    except (TypeError, ValueError):
        return 0


def golden_files() -> list[pathlib.Path]:
    return sorted(GOLDEN.glob("*.json")) if GOLDEN.exists() else []


def load_golden() -> list[dict]:
    out = []
    for p in golden_files():
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return out


def measure_pair() -> list[dict]:
    return [g for g in load_golden() if g.get("measure_pair") is True]


def parse_junit(path: pathlib.Path) -> dict:
    """Counts from pytest's OWN junit report, never from prose."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {"collected": 0, "errors": 1, "passed": 0, "failed": 0, "skipped": 0}
    tests = int(suite.get("tests", "0"))
    failures = int(suite.get("failures", "0"))
    errors = int(suite.get("errors", "0"))
    skipped = int(suite.get("skipped", "0"))
    return {
        "collected": tests,
        "errors": errors,
        "passed": tests - failures - errors - skipped,
        "failed": failures,
        "skipped": skipped,
    }


def run_pytest(target: pathlib.Path, junit: pathlib.Path) -> dict:
    junit.parent.mkdir(parents=True, exist_ok=True)
    if junit.exists():
        junit.unlink()
    cmd = [str(VENV_PY), "-m", "pytest", str(target), "-c", str(PYTEST_INI),
           "-q", "--junitxml", str(junit)]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), timeout=1800)
    if not junit.exists():
        return {"collected": 0, "errors": 1, "passed": 0, "failed": 0, "skipped": 0}
    return parse_junit(junit)


def write_json(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(repo: pathlib.Path, *args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(repo),
                       timeout=120)
    return r.stdout.strip()
