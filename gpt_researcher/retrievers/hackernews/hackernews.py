# Hacker News Retriever
#
# Searches Hacker News stories via the free Algolia HN Search API.
# No API key required. Good for developer/tech trend signal and discussion.
# API docs: https://hn.algolia.com/api

import json
import logging
import urllib.parse
import urllib.request


logger = logging.getLogger(__name__)


class HackerNewsSearch:
    """
    Hacker News search retriever (Algolia HN Search API, keyless).

    Returns results in the standard {title, href, body} format used by all
    GPT Researcher retrievers. Pass by_date=True to sort by recency instead of
    relevance (useful for "what's trending recently" queries).
    """

    def __init__(self, query, query_domains=None, by_date=False, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.by_date = bool(by_date)

    def search(self, max_results=7):
        """
        Search Hacker News via the Algolia API.

        Returns:
            list: Search results as [{title, href, body}, ...]
        """
        print(f"Searching Hacker News with query: {self.query}...")
        try:
            return self._search(max_results)
        except Exception as e:
            logger.error(f"Error: {e}. Failed fetching Hacker News sources. Resulting in empty response.")
            return []

    def _search(self, max_results):
        endpoint = "search_by_date" if self.by_date else "search"
        params = urllib.parse.urlencode({
            "query": self.query,
            "tags": "story",
            "hitsPerPage": min(max_results, 50),
        })
        url = f"https://hn.algolia.com/api/v1/{endpoint}?{params}"

        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "gpt-researcher/1.0",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for hit in data.get("hits", [])[:max_results]:
            obj_id = hit.get("objectID", "")
            discussion = f"https://news.ycombinator.com/item?id={obj_id}"
            title = hit.get("title") or hit.get("story_title") or "(untitled)"
            href = hit.get("url") or discussion
            points = hit.get("points") or 0
            comments = hit.get("num_comments") or 0
            author = hit.get("author") or "unknown"
            text = (hit.get("story_text") or "").strip()

            body = (
                f"{title}\n"
                f"[Hacker News: {points} points, {comments} comments by {author}] "
                f"Discussion: {discussion}"
            )
            if text:
                body += f"\n\n{text[:800]}"

            results.append({"title": title, "href": href, "body": body})
        return results
