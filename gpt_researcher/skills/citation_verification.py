"""CitationAgent — post-pass citation verification for deep research.

Re-fetches each learning's citation URL via firecrawl /v2/scrape with
maxAge=0 (bypass cache) and checks the cited text against the fetched
markdown. Unmatched or unfetchable claims are flagged unverified.
"""
from __future__ import annotations

import os
import re
import time

import requests

_FIRECRAWL_V2_SCRAPE = "https://api.firecrawl.dev/v2/scrape"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_supported(quote: str, source: str) -> bool:
    """True if source contains quote or shares >=0.7 of its significant words."""
    q, s = _normalize(quote), _normalize(source)
    if q and q in s:
        return True
    # ponytail: learnings are paraphrased live, so exact containment is too
    # strict there — fall back to significant-word overlap >= 0.7.
    # A word also counts if its crude stem (drop last 2 chars) appears,
    # so "providers"/"provider", "regulations"/"regulation" still match.
    words = {w for w in re.findall(r"[a-z0-9]{4,}", q)}
    if len(words) < 4:
        # a 1-3-word claim clears 0.7 on coincidental vocabulary against almost
        # any prose — short claims must match verbatim, never by overlap
        return False
    hits = sum(1 for w in words if w in s or (len(w) > 5 and w[:-2] in s))
    return hits / len(words) >= 0.7


class CitationAgent:
    """Verify {quote/learning: citation url} claims against live sources."""

    def __init__(self, timeout: int = 60) -> None:
        self.api_key = os.getenv("FIRECRAWL_API_KEY", "")
        self.timeout = timeout

    def _scrape(self, url: str) -> str | None:
        """Fetch fresh markdown for url; None if unfetchable.

        Retries transient failures (429/5xx, network errors) so one rate-limit
        blip does not mark every claim citing that URL unverified.
        """
        for attempt in range(3):
            try:
                resp = requests.post(
                    _FIRECRAWL_V2_SCRAPE,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"url": url, "maxAge": 0, "formats": ["markdown"]},
                    timeout=self.timeout,
                )
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not data.get("success"):
                    return None
                return data.get("data", {}).get("markdown") or None
            except Exception:
                time.sleep(1)
        return None

    def _matches(self, quote: str, source: str) -> bool:
        return text_supported(quote, source)

    def verify(self, citations: dict[str, str],
               documents: dict[str, str] | None = None) -> dict:
        """Verify {quote: url} claims; documents (url -> already-read text) seeds
        the fetch cache so those URLs are checked locally instead of re-scraped."""
        claims = []
        cache: dict[str, str | None] = dict(documents) if documents else {}
        for quote, url in citations.items():
            if url not in cache:
                cache[url] = self._scrape(url) if url else None
            source = cache[url]
            verified = bool(source) and self._matches(quote, source)
            claims.append({"quote": quote, "url": url, "verified": verified})
        grounded = sum(1 for c in claims if c["verified"])
        return {
            "total_claims": len(claims),
            "grounded": grounded,
            "unverified": len(claims) - grounded,
            "claims": claims,
        }
