"""Content extraction through a locally hosted Crawl4AI server.

Why this exists. The two scrapers that shipped before it each fail on a different half
of the web: ``bs`` (BeautifulSoup over ``requests``) cannot run JavaScript, so a
client-rendered page yields a shell with no article in it, and ``firecrawl`` bills a
credit per page against a quota that a research run shares with every other caller of
the same key -- measured 2026-09-20, the 1,000/month free tier was at 0 with ten days
left in the period. Crawl4AI runs Chromium in a container on this host: it renders the
page like the browser scraper does, costs nothing per page, and has no quota to run out.

It is NOT a replacement for Firecrawl's SEARCH retriever. Crawl4AI is handed a URL and
returns that page; it has no search endpoint. Web search stays with Tavily/Serper/DDG.

The endpoint is ``/crawl`` rather than the simpler ``/md`` because the scraper contract
here is ``(content, images, title)``: ``/md`` returns markdown alone, so the title would
have to be guessed from the first heading and the images would need a second fetch of
the same page -- which is what the Firecrawl scraper does, and it doubles the work per
URL. One ``/crawl`` carries all three.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://crawl4ai:11235"

# Chromium has to start, navigate, and settle before any markdown exists. Measured
# against the container on this host: a static page answers in 2-4s, a JS-heavy one in
# 10-20s. The ceiling is generous because the cost of cutting a slow page off is a
# source lost from the report, while the cost of waiting is one slot in a thread pool
# that is already bounded by MAX_SCRAPER_WORKERS.
DEFAULT_TIMEOUT_S = float(os.getenv("CRAWL4AI_TIMEOUT_S", "60"))


class Crawl4AIScraper:
    """Scraper backed by the Crawl4AI REST API."""

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        self.base_url = os.getenv("CRAWL4AI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        # Set whenever the server is reachable at all: 0.9.3 binds its own loopback
        # unless CRAWL4AI_API_TOKEN is set, and setting it turns Bearer auth on for
        # every endpoint. Absent, requests go out unauthenticated -- which is correct
        # for an older server that never required a token.
        self.token = os.getenv("CRAWL4AI_API_TOKEN", "")

    def scrape(self) -> tuple:
        """``(content, image_urls, title)``, or ``("", [], "")`` on any failure.

        Never raises. Every caller of this package treats an empty body as "this source
        gave nothing" and moves on, and a raising scraper takes the whole batch of URLs
        down with it -- see `Scraper.extract_data_from_url`, which gathers results per
        URL and has no per-URL recovery of its own.
        """
        try:
            response = requests.post(
                f"{self.base_url}/crawl",
                json={"urls": [self.link]},
                headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                timeout=DEFAULT_TIMEOUT_S,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(f"Crawl4AI request failed for {self.link}: {exc}")
            return "", [], ""

        result = self._first_result(payload)
        if result is None:
            logger.warning(f"Crawl4AI returned no result for {self.link}")
            return "", [], ""

        # `success: false` still carries a 200 and a result object; the body is empty or
        # an error page, and returning it would put a Cloudflare notice into the report
        # as though it were the source.
        if result.get("success") is False:
            logger.warning(f"Crawl4AI could not render {self.link}: "
                           f"{result.get('error_message') or 'no error given'}")
            return "", [], ""

        content = self._markdown(result)
        title = (result.get("metadata") or {}).get("title") or ""
        images = self._images(result)
        return content, images, title

    @staticmethod
    def _first_result(payload):
        """The single result, from whichever envelope this server version uses.

        The REST API has shipped the crawl under `results`, and a bare object for a
        one-URL request; both appear in the wild depending on version, so neither is
        assumed.
        """
        if isinstance(payload, list):
            return payload[0] if payload else None
        if not isinstance(payload, dict):
            return None
        for key in ("results", "result", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value[0] if value else None
            if isinstance(value, dict):
                return value
        return payload if "markdown" in payload else None

    @staticmethod
    def _markdown(result) -> str:
        """Markdown as a string, whether the server sends a string or the object.

        Newer versions send `{raw_markdown, fit_markdown, ...}`. `fit_markdown` is the
        filtered variant and is preferred when present -- it is what drops navigation
        and boilerplate -- but it is empty when no content filter is configured, which
        is the default, so raw is the fallback rather than the other way round.
        """
        markdown = result.get("markdown")
        if isinstance(markdown, str):
            return markdown
        if isinstance(markdown, dict):
            return (markdown.get("fit_markdown")
                    or markdown.get("raw_markdown")
                    or markdown.get("markdown_with_citations")
                    or "")
        return result.get("cleaned_html") or ""

    @staticmethod
    def _images(result) -> list:
        """``[{"url": ..., "score": ...}]``, the shape `get_relevant_images` returns.

        NOT a list of URLs, which is what this returned when it was written. The
        consumers index the entries: `skills/browser.py` sorts on `im["score"]` and reads
        `img["url"]`, and `actions/report_generation.py` reads `img['url']` -- so a list
        of strings raises `TypeError: string indices must be integers` inside the scrape
        loop, every scraped page dies, and the node fails with "0 context chars" while
        the retrievers report documents read. Measured 2026-09-20: one tree run,
        8 documents read, empty report.

        Crawl4AI scores its own images for relevance, so that score is carried through
        rather than re-derived; entries without a `src` are decorative and drop out. Ten
        is the same ceiling `get_relevant_images` applies.
        """
        media = result.get("media")
        if not isinstance(media, dict):
            return []
        images = media.get("images")
        if not isinstance(images, list):
            return []
        out = [{"url": img["src"], "score": img.get("score") or 0}
               for img in images if isinstance(img, dict) and img.get("src")]
        out.sort(key=lambda im: im["score"], reverse=True)
        return out[:10]
