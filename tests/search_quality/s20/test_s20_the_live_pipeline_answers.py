"""One real research call, through the MCP endpoint, asserted against real failure modes.

Every other test in this suite is offline: it mocks the retriever, the scraper or the
model and checks one seam. That is the right default -- they are fast and free -- but it
is also how this pipeline broke twice in one day while the suite stayed green:

- 2026-09-20, the crawl4ai scraper returned image URLs where the consumers index dicts.
  Every scrape raised inside the per-document loop. The retrievers still reported
  documents read, the tree still reported `researched: 1`, and the report on disk was the
  17 bytes `_(no synthesis)_`. 16 unit tests passed throughout, because they asserted the
  scraper's own output rather than what the consumer does with it.
- The same day, the LLM provider hit a weekly limit. `deep_tree_research` returned
  `status: "success"` with `nodes_researched: 0` and an empty report.

What those have in common is that the CALL SUCCEEDED. Nothing raised, the status field
said success, and the artifact was empty. No amount of mocking finds that, because the
mocks are exactly what the real components stopped doing. So this test runs the whole
thing -- SearXNG, the scraper container, the model, synthesis -- and asserts on evidence
that only exists when work actually happened.

It costs money and about two minutes, so it does not run by default. Turn it on with:

    GPTR_E2E=1 venv/Scripts/python -m pytest tests/search_quality/s20 -c harness-search/pytest.ini -s

Run it after anything that touches the retrievers, the scrapers, the LLM provider or the
container wiring. That is when a green offline suite is least informative.
"""
import json
import os
import re
import urllib.error
import urllib.request

import pytest

ENDPOINT = os.getenv("GPTR_MCP_URL", "http://127.0.0.1:8123/mcp")
HEALTH = ENDPOINT.rsplit("/", 1)[0] + "/health"

# A question with a real literature behind it, narrow enough for one node, and stable
# enough that "no sources exist" is never the explanation for a thin result.
QUERY = ("What are the practical tradeoffs of using PostgreSQL logical replication "
         "for major version upgrades?")

# The report the tree writes when no node produced an answer. Matching the marker is
# better than a length threshold: it is the exact artifact the two real failures left.
NO_SYNTHESIS = "_(no synthesis)_"


def _sse_payload(body: bytes) -> dict:
    """The JSON out of an SSE response. FastMCP answers `event:`/`data:` lines even for
    a single result, so the plain `json.loads(body)` a REST client would write fails."""
    for line in body.decode("utf-8", "replace").splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise AssertionError(f"no SSE data line in: {body[:200]!r}")


def _post(payload: dict, session: str | None, timeout: float) -> tuple[dict, str | None]:
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session:
        headers["mcp-session-id"] = session
    request = urllib.request.Request(ENDPOINT, json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        returned = response.headers.get("mcp-session-id") or session
    return (_sse_payload(body) if body.strip() else {}), returned


def _call_tool(name: str, arguments: dict, timeout: float) -> dict:
    """Initialize, notify, call -- the handshake a real MCP client performs."""
    hello, session = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "s20-e2e", "version": "1"}}},
                           None, 30)
    assert "result" in hello, f"initialize failed: {hello}"
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session, 30)

    reply, _ = _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": name, "arguments": arguments}}, session, timeout)
    assert "error" not in reply, f"{name} returned an error: {reply['error']}"
    content = reply["result"]["content"][0]["text"]
    return json.loads(content)


@pytest.fixture(scope="module")
def live_result():
    if not os.getenv("GPTR_E2E"):
        pytest.skip("live pipeline test is opt-in; set GPTR_E2E=1 (costs tokens + ~2min)")
    try:
        urllib.request.urlopen(HEALTH, timeout=10).read()
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"gptr-mcp not reachable at {HEALTH} ({exc})")

    # light: one node. Enough to prove search, scrape, answer and synthesis all ran,
    # without paying for a tree whose branches this test does not assert on.
    return _call_tool("deep_tree_research", {"query": QUERY, "depth": "light"}, 420)


# ------------------------------------------------------------- the call itself

def test_the_provider_answered(live_result):
    """A refused provider returns `status: success` with nothing researched. That is the
    2026-09-20 weekly-limit failure, and it must read as a failed test rather than as a
    thin report."""
    stats = live_result["stats"]
    assert not stats.get("provider_refused"), (
        f"the LLM provider refused: {stats.get('provider_refusal_reason')!r} -- "
        "nothing was researched, so nothing below is meaningful")


def test_a_node_was_actually_researched(live_result):
    """`researched` counts nodes that produced an answer, not nodes attempted."""
    stats = live_result["stats"]
    assert stats["nodes_researched"] >= 1, (
        f"0 of {stats['nodes_total']} nodes researched "
        f"(pending {stats.get('pending_count')}, time budget exhausted "
        f"{stats.get('time_budget_exhausted')}) -- search or scrape produced no evidence")


def test_the_report_is_not_the_empty_marker(live_result):
    """The exact artifact both real failures left on disk."""
    path = live_result["report_path"]
    assert os.path.exists(path), f"report_path does not exist: {path}"
    report = open(path, encoding="utf-8").read()

    assert NO_SYNTHESIS not in report, (
        f"the tree wrote {NO_SYNTHESIS!r} to {path}: nodes ran but none yielded an answer")
    assert len(report) > 1000, f"report is {len(report)} chars, which is not a report: {path}"


def test_the_report_cites_its_sources(live_result):
    """Citations are the end-to-end signal: a claim only carries one when a retriever
    found the page, the scraper read it, and synthesis kept the link intact. The s18
    scraper bug showed up here as 0 while everything upstream looked healthy."""
    assert live_result["citation_count"] >= 1, (
        "the report cites nothing -- scraping or citation mapping dropped every source")

    report = open(live_result["report_path"], encoding="utf-8").read()
    assert re.search(r"https?://", report), "no source URL survived into the report body"


def test_synthesis_kept_most_of_what_the_nodes_found(live_result):
    """`rollup_dropped_ratio` is the share of the body the roll-up discarded as
    unsupported by any node answer. Near 1.0 means the nodes failed and synthesis threw
    the rest away -- measured 0.91 on the empty run, 0.0 on the healthy one."""
    dropped = live_result["stats"].get("rollup_dropped_ratio", 0)
    assert dropped < 0.9, (
        f"the roll-up dropped {dropped:.0%} of the body as unsupported")


def test_the_scrapers_produced_text_not_just_documents(live_result):
    """The shape of the s18 failure: documents read, zero context. Claim units only
    exist when scraped bodies reached the merge stage with content in them."""
    merge = live_result["stats"].get("merge", {})
    assert merge.get("chars_before", 0) > 0 or merge.get("claim_units", 0) > 0, (
        f"nothing reached the merge stage ({merge}) -- pages were fetched but produced "
        "no text, which is what a scraper returning the wrong shape looks like")
