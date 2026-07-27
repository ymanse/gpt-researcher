"""s1 retrieval probe — copied INTO the container by measure.py and executed there:

    docker cp scripts/container_probe_s1.py gptr-mcp-server:/tmp/probe_s1.py
    docker exec gptr-mcp-server python /tmp/probe_s1.py "<query_id>" "<query text>"

Runs GPTResearcher.conduct_research() directly and introspects the researcher object —
no implementation-side logging contract needed. Counts retriever/scraper failures by
capturing ERROR-level log records and retriever exceptions. Prints exactly one line:

    SQ_PROBE {"query_id":..,"scraped_pages":N,"context_chars":N,"retriever_errors":N}
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys


UNRECOVERED = "SQ_RETRIEVAL_UNRECOVERED"


class _ErrCounter(logging.Handler):
    """Counts UNRECOVERED retrieval failures — not every ERROR in the process.

    The first version attached to the root logger and counted every ERROR record, so an
    adapter's own internal error for a failure a SIBLING retriever had already covered
    (tavily 432 with firecrawl returning fine) tripped the gate. That made
    retriever_errors_total == 0 a function of external service weather rather than of the
    code (s1 review R1). Per spec s1 we now count only records carrying the
    implementation's explicit unrecovered marker; the raw count is still reported as
    error_records_total for visibility, but nothing gates on it.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.unrecovered = 0
        self.total = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.total += 1
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — a broken formatter must not kill the probe
            msg = str(record.msg)
        if UNRECOVERED in msg:
            self.unrecovered += 1


def _marker_wired(pkg) -> bool:
    """law 4: a marker nobody emits makes retriever_errors a hollow zero. Prove the
    implementation actually contains the marker before believing a count of 0."""
    import pathlib
    root = pathlib.Path(pkg.__file__).parent
    for p in root.rglob("*.py"):
        try:
            if UNRECOVERED in p.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


async def main() -> int:
    qid, query = sys.argv[1], sys.argv[2]
    counter = _ErrCounter()
    logging.getLogger().addHandler(counter)

    import gpt_researcher
    from gpt_researcher import GPTResearcher
    wired = _marker_wired(gpt_researcher)
    r = GPTResearcher(query=query, report_type="research_report")
    await r.conduct_research()

    visited = getattr(r, "visited_urls", None) or set()
    ctx = getattr(r, "context", "") or ""
    if isinstance(ctx, list):
        ctx = "\n".join(str(c) for c in ctx)
    print("SQ_PROBE " + json.dumps({
        "query_id": qid,
        "scraped_pages": len(visited),
        "context_chars": len(ctx),
        "retriever_errors": counter.unrecovered,
        "error_records_total": counter.total,
        "marker_wired": wired,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
