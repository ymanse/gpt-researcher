# Reddit Retriever (via Firecrawl)
#
# Reddit blocks datacenter IPs at the CDN level (every host — www/old/oauth/api —
# returns a 403 challenge page), so direct access from a server is impossible.
# We fetch Reddit discussions through the Firecrawl Search API, whose
# infrastructure reaches Reddit fine. Requires FIRECRAWL_API_KEY (free tier at
# https://firecrawl.dev). Without the key the retriever degrades to [] (never raises).

import json
import logging
import os
import urllib.request

_FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v1/search"


logger = logging.getLogger(__name__)


class RedditSearch:
    """
    Reddit search retriever via the Firecrawl Search API (scoped to reddit.com).

    Set FIRECRAWL_API_KEY in the env. Returns the standard {title, href, body}
    shape. (Direct Reddit is CDN-blocked from datacenter IPs; Firecrawl bypasses it.)
    """

    def __init__(self, query, query_domains=None, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.api_key = os.getenv("FIRECRAWL_API_KEY")

    def search(self, max_results=7):
        print(f"Searching Reddit (via Firecrawl) with query: {self.query}...")
        if not self.api_key:
            print("Reddit: no FIRECRAWL_API_KEY set — skipping.")
            return []
        try:
            return self._search(max_results)
        except Exception as e:
            logger.error(f"Error: {e}. Failed fetching Reddit sources. Resulting in empty response.")
            return []

    def _search(self, max_results):
        payload = json.dumps({
            "query": f"{self.query} site:reddit.com",
            "limit": min(max(max_results, 1), 20),
        }).encode("utf-8")

        req = urllib.request.Request(
            _FIRECRAWL_SEARCH,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "gpt-researcher/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Firecrawl search response: {"success": true, "data": [ {url,title,description,markdown?} ]}
        # Some versions nest results under data.web — handle both.
        items = data.get("data")
        if isinstance(items, dict):
            items = items.get("web") or items.get("results") or []
        items = items or []

        results = []
        for it in items[:max_results]:
            url = it.get("url", "")
            if "reddit.com" not in url:
                continue
            title = it.get("title") or "(untitled)"
            body = (it.get("markdown") or it.get("description") or it.get("snippet") or "").strip()
            results.append({"title": title, "href": url, "body": body})
        return results
