"""Shared constants + helpers for the tier-a harness scripts.

Every evidence file the gates read is produced by these scripts from a real tool's own
output (pytest junit xml, docker logs, git) — never hand-authored (law 10). All emitters
are idempotent: same inputs, same bytes.
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
TESTS = REPO / "tests" / "tier_a"
PYTEST_INI = HARNESS / "pytest.ini"
COMPOSE_FILE = "D:/docker/gptr-mcp/docker-compose.yml"
COMPOSE_SERVICE = "gptr-mcp"
CONTAINER = "gptr-mcp-server"
MCP_URL = "http://127.0.0.1:8123/mcp"
HEALTH_URL = "http://127.0.0.1:8123/health"
OUTPUTS_HOST = MCP_REPO / "outputs"
BRANCH = "feature/tier-a-upgrade"

SMOKE_QUERIES = {
    1: "latest developments in solid-state battery manufacturing 2026",
    2: "impact of EU AI Act enforcement on open-source model providers",
    3: "impact of EU AI Act enforcement on open-source model providers",
    4: "sparse attention long context transformer efficiency survey 2025..2026 recent papers",
    5: "impact of EU AI Act enforcement on open-source model providers",
    6: "solid-state battery supply chain: manufacturing bottlenecks and key players",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if not junit.exists():
        return {"collected": 0, "errors": 1, "passed": 0, "failed": 0, "skipped": 0}
    return parse_junit(junit)


def write_json(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def git(repo: pathlib.Path, *args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(repo))
    return r.stdout.strip()
