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

from ..scraper.utils import detect_unreadable

_FIRECRAWL_V2_SCRAPE = "https://api.firecrawl.dev/v2/scrape"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# a paraphrased claim must trace to ONE passage this many tokens wide (~2-4
# sentences); claim words scattered wider than this across the document are
# topical co-occurrence, not quotation
_PASSAGE_TOKENS = 60


def _passage_covers(words: set[str], source: str) -> bool:
    """True if one _PASSAGE_TOKENS-token window of source matches >=0.7 of words.

    A source token matches a claim word exactly or by crude stem (claim word
    minus its last 2 chars), so "providers"/"provider", "regulations"/
    "regulation" still align.
    """
    stems = {w[:-2]: w for w in words if len(w) > 5}
    hits: list[tuple[int, str]] = []  # (token position, claim word matched)
    for i, tok in enumerate(re.findall(r"[a-z0-9]+", source)):
        if tok in words:
            hits.append((i, tok))
            continue
        # ponytail: linear stem scan per token; bucket stems by length if docs grow
        for stem, w in stems.items():
            if tok.startswith(stem):
                hits.append((i, w))
                break
    counts: dict[str, int] = {}
    lo = 0
    for pos, w in hits:
        counts[w] = counts.get(w, 0) + 1
        while pos - hits[lo][0] >= _PASSAGE_TOKENS:
            lw = hits[lo][1]
            counts[lw] -= 1
            if not counts[lw]:
                del counts[lw]
            lo += 1
        if len(counts) / len(words) >= 0.7:
            return True
    return False


def text_supported(quote: str, source: str) -> bool:
    """True if source contains quote verbatim, or one bounded passage of source
    covers >=0.7 of the quote's significant words."""
    q, s = _normalize(quote), _normalize(source)
    if q and q in s:
        return True
    words = {w for w in re.findall(r"[a-z0-9]{4,}", q)}
    if len(words) < 4:
        # a 1-3-word claim clears 0.7 on coincidental vocabulary against almost
        # any prose — short claims must match verbatim, never by overlap
        return False
    # ponytail: learnings are paraphrased live, so exact containment is too
    # strict — but whole-document word overlap mistakes a topically-similar
    # unused source for a quoted one: genuine (para)quotation comes from a
    # specific passage, so the overlap must concentrate in one window.
    return _passage_covers(words, s)


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
                md = data.get("data", {}).get("markdown") or None
                # A wall is not a page here either. This is the ONE seam that decides
                # whether a claim reads as "we never got the source" or "the source does
                # not back it", so accepting any non-empty markdown means an
                # interstitial re-fetched here becomes a source that REFUTES the claim.
                # Same detector the scraper uses, so the two lanes agree on what a page
                # is. No html and no status to offer it — the markdown is all we have,
                # which is why the machine tokens and the short-document banner rule
                # both still apply.
                if md and detect_unreadable(md, md):
                    return None
                return md
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
            claim = {"quote": quote, "url": url, "verified": verified}
            # WHY it failed, on the claim itself. `bool(source)` False means we never
            # obtained the page — blocked, rate-limited, no url at all — while _matches
            # False means we read the page and it does not carry this quote. Those are
            # opposite findings: the first is a hole in the evidence, the second is
            # evidence against the claim. Collapsing them is what let a correct figure
            # be discarded because the source it came from had been silently walled off
            # (preprints.org, 2026-08-04).
            #
            # A SUB-PARTITION, not a third bucket: total_claims / grounded / unverified
            # keep their meanings and their arithmetic, and an unretrieved claim stays
            # counted in `unverified` — pinned by tests/tier_a/stage3/test_stage3_citation.py
            # (:126 requires unverified==1 for an unfetchable url, :151 requires
            # total == grounded + unverified, :157 asserts exact equality on the empty
            # result, so no new top-level key may appear).
            #
            # `url and` is load-bearing. An EMPTY url does not mean "we failed to fetch
            # it" — it means the caller already looked and found no source of its own
            # that supports the claim. tree_research builds claim_urls exactly that way
            # (`next((u for u in n.sources if text_supported(...)), "")`), so without
            # this guard the tree path would stamp `unretrieved` on the one case that is
            # the OPPOSITE of a retrieval hole: every source was read, and none of them
            # carried the claim. That is evidence, and mislabelling it as a gap is the
            # same inversion this flag exists to prevent.
            if not verified and url and not source:
                claim["unretrieved"] = True
            claims.append(claim)
        grounded = sum(1 for c in claims if c["verified"])
        return {
            "total_claims": len(claims),
            "grounded": grounded,
            "unverified": len(claims) - grounded,
            "claims": claims,
        }
