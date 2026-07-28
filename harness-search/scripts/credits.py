"""Firecrawl credit meter — the offline loop's cost proof.

    python scripts/credits.py          -> credits_remaining=38355  (or credits_remaining=-1)

The dedup harness's whole premise is that re-synthesis is free: it replays cached node
answers, so it must not buy a single search. "No retrieval calls" is easy to claim and
hard to see; the team credit balance is the one number the vendor keeps for us, so the
offline gate reads it before and after and requires the delta to be zero.

-1 means "could not read the balance" and every caller treats that as a FAIL, never as
zero spend (law 4: a number that means 'no data' may not be cited as proof).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.error
import urllib.request

ENV_FILE = pathlib.Path("D:/docker/gptr-mcp/.env")
URL = "https://api.firecrawl.dev/v2/team/credit-usage"


def api_key() -> str:
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if key:
        return key
    try:
        for ln in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*FIRECRAWL_API_KEY\s*=\s*(.+?)\s*$", ln)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def remaining() -> int:
    key = api_key()
    if not key:
        return -1
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return -1
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, dict):
        for k in ("remaining_credits", "remainingCredits", "credits_remaining"):
            if isinstance(data.get(k), (int, float)):
                return int(data[k])
    return -1


if __name__ == "__main__":
    print(f"credits_remaining={remaining()}")
