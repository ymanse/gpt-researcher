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


class _ErrCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


async def main() -> int:
    qid, query = sys.argv[1], sys.argv[2]
    counter = _ErrCounter()
    logging.getLogger().addHandler(counter)

    from gpt_researcher import GPTResearcher
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
        "retriever_errors": counter.count,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
