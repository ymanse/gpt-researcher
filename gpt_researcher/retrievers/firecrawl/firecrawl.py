# Firecrawl Search Retriever
#
# Calls Firecrawl /v2/search with scrapeOptions.formats=["markdown"] so every
# web result carries the full-page clean markdown (not just a snippet).
# Requires FIRECRAWL_API_KEY (https://firecrawl.dev).

import logging
import os

import requests

_FIRECRAWL_V2_SEARCH = "https://api.firecrawl.dev/v2/search"


logger = logging.getLogger(__name__)


class FirecrawlSearch:
    """
    Firecrawl /v2/search retriever returning [{href, title, body}] where body
    is the scraped full-page markdown.
    """

    def __init__(self, query, query_domains=None, categories=None, tbs=None, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.categories = categories
        self.tbs = tbs
        self.api_key = os.getenv("FIRECRAWL_API_KEY")

    def search(self, max_results=7):
        print(f"Searching with query {self.query}...")
        if not self.api_key:
            print("Firecrawl: no FIRECRAWL_API_KEY set — skipping.")
            return []
        try:
            return self._search(max_results)
        except Exception as e:
            logger.error(f"Error: {e}. Failed fetching sources from Firecrawl. Resulting in empty response.")
            return []

    def _search(self, max_results):
        query = self.query
        if self.query_domains:
            query += " " + " OR ".join(f"site:{d}" for d in self.query_domains)

        payload = {
            "query": query,
            "limit": min(max(max_results, 1), 20),
            "scrapeOptions": {"formats": ["markdown"]},
        }
        if self.categories:
            payload["categories"] = self.categories
        if self.tbs:
            payload["tbs"] = self.tbs

        resp = requests.post(
            _FIRECRAWL_V2_SEARCH,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("data") or {}
        if isinstance(items, dict):
            items = items.get("web") or items.get("results") or []

        results = []
        for it in items[:max_results]:
            href = it.get("url", "")
            body = (it.get("markdown") or "").strip()
            if not href or not body:
                continue
            results.append({
                "href": href,
                "title": it.get("title") or "(untitled)",
                "body": body,
                # Firecrawl already scraped the full page — mark it so the research
                # pipeline doesn't re-scrape (see researcher.py raw_content check).
                "raw_content": body,
            })

        max_body_len = max((len(r["body"]) for r in results), default=0)
        print(f"TIERA_EVIDENCE stage=1 firecrawl_results={len(results)} max_body_len={max_body_len}", flush=True)
        return results
