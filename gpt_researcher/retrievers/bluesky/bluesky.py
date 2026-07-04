# Bluesky Retriever
#
# Searches Bluesky posts via the public AT Protocol AppView
# (app.bsky.feed.searchPosts). No API key / login required for public search.
# API: https://api.bsky.app/xrpc/app.bsky.feed.searchPosts

import json
import urllib.parse
import urllib.request


class BlueskySearch:
    """
    Bluesky (AT Protocol) post search retriever — keyless.

    Uses the public AppView searchPosts endpoint. Returns the standard
    {title, href, body} shape. sort="top" (default) ranks by engagement,
    "latest" by recency (useful for trend queries).
    """

    def __init__(self, query, query_domains=None, sort="top", **kwargs):
        self.query = query
        self.query_domains = query_domains
        self.sort = sort if sort in ("top", "latest") else "top"

    def search(self, max_results=7):
        """
        Search Bluesky via the public AppView.

        Returns:
            list: Search results as [{title, href, body}, ...]
        """
        print(f"Searching Bluesky with query: {self.query}...")
        try:
            return self._search(max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching Bluesky sources. Resulting in empty response.")
            return []

    def _search(self, max_results):
        params = urllib.parse.urlencode({
            "q": self.query,
            "limit": min(max(max_results, 1), 100),
            "sort": self.sort,
        })
        # ponytail: api.bsky.app (NOT public.api.bsky.app) — the "public" host
        # WAF-403s datacenter IPs; api.bsky.app serves searchPosts unauthenticated.
        url = f"https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?{params}"

        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "gpt-researcher/1.0",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for post in data.get("posts", [])[:max_results]:
            author = post.get("author", {})
            handle = author.get("handle", "unknown")
            name = author.get("displayName") or handle
            text = (post.get("record", {}).get("text") or "").strip()

            likes = post.get("likeCount", 0)
            reposts = post.get("repostCount", 0)
            replies = post.get("replyCount", 0)

            # at://did/app.bsky.feed.post/<rkey> -> bsky.app/profile/<handle>/post/<rkey>
            uri = post.get("uri", "")
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            href = (
                f"https://bsky.app/profile/{handle}/post/{rkey}"
                if rkey else f"https://bsky.app/profile/{handle}"
            )

            title = f"@{handle}: {text[:120]}{'...' if len(text) > 120 else ''}"
            body = (
                f"{name} (@{handle}) on Bluesky:\n{text}\n\n"
                f"[{likes} likes, {reposts} reposts, {replies} replies]"
            )
            results.append({"title": title, "href": href, "body": body})
        return results
