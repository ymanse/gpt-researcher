# GitHub Retriever
#
# "Trending" proxy via the GitHub Search API: repositories matching the query,
# sorted by stars (most-starred relevant repos = the GitHub signal for a topic).
# Keyless (10 req/min); set GITHUB_TOKEN to raise the limit to 30 req/min.
# API: https://api.github.com/search/repositories

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request


logger = logging.getLogger(__name__)


class GithubSearch:
    """
    GitHub repository search retriever (Search API, stars-sorted = trend signal).

    Keyless by default; set GITHUB_TOKEN to raise the rate limit. Returns the
    standard {title, href, body} shape.
    """

    def __init__(self, query, query_domains=None, **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.token = os.getenv("GITHUB_TOKEN")

    def search(self, max_results=7):
        print(f"Searching GitHub with query: {self.query}...")
        try:
            return self._search(max_results)
        except urllib.error.HTTPError:
            # HTTP failures (e.g. 422 malformed request) must surface so the caller
            # (SmartRetriever) can classify retry-worthiness and route around a
            # retriever that cannot serve — swallowed, they read as "no results" and
            # get re-paid for on every subsequent sub-query.
            raise
        except Exception as e:
            logger.error(f"Error: {e}. Failed fetching GitHub sources. Resulting in empty response.")
            return []

    def _search(self, max_results):
        params = urllib.parse.urlencode({
            "q": self.query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max(max_results, 1), 30),
        })
        url = f"https://api.github.com/search/repositories?{params}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "gpt-researcher/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for repo in data.get("items", [])[:max_results]:
            name = repo.get("full_name", "")
            href = repo.get("html_url", "")
            desc = (repo.get("description") or "").strip()
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "?"
            topics = ", ".join(repo.get("topics", [])[:6])
            title = f"{name} (★{stars})"
            body = (
                f"{name} - {desc}\n"
                f"[★{stars} stars | {lang}"
                + (f" | topics: {topics}" if topics else "")
                + f"] {href}"
            )
            results.append({"title": title, "href": href, "body": body})
        return results
