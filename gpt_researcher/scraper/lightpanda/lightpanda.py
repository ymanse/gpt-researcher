"""Render a page in Lightpanda, and fall back to Chromium when it comes back thin.

Lightpanda is a headless browser written in Zig that runs V8 and the DOM but implements
no layout, paint or compositing. That is the whole trade: it costs ~5MB resident against
Chromium's ~430MB and renders about three times faster, but anything that needs a
rendering surface -- canvas, WebGL, fingerprint-based anti-bot, some hydration paths --
gets nothing out of it.

Measured on this host, 24 URLs taken from real research runs, both engines rendered and
both extracted with the same extractor: 21 matched Chromium's text within 20%, and three
did not (stackoverflow.com 2170 -> 41 chars, materializedview.io 15842 -> 170,
qualcomm.com 7207 -> 1456). Median 0.8s against 2.2s.

So the useful shape is not "replace Chromium" but "try the cheap engine, keep the
expensive one for the pages that need it" -- which is also what the research on
production deployments recommends. This class is that router: Lightpanda first, and
`Crawl4AIScraper` (Chromium, in its own container) whenever Lightpanda errors or returns
less text than a real article would have.

WHY IT CONNECTS BY IP. Lightpanda rejects a WebSocket whose Host header is a name it
does not recognise -- DNS-rebinding protection -- and answers 403 "Host not allowed".
`ws://lightpanda:9222/` is exactly that case, which is what stopped Crawl4AI's own
`cdp_url` from ever working here. Resolving the service name ourselves and dialling the
address keeps the compose service name in config (container IPs are not stable) while
sending a Host header Lightpanda accepts.
"""
import logging
import os
import socket
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

from ..utils import (
    clean_soup,
    detect_unreadable,
    extract_title,
    get_relevant_images,
    get_text_from_soup,
)

logger = logging.getLogger(__name__)

DEFAULT_CDP_URL = "ws://lightpanda:9222/"

# Below this many characters the page is treated as un-rendered and Chromium is asked.
# Chosen from the measured failures: the two hard ones returned 41 and 170 characters,
# while the thinnest page Lightpanda rendered CORRECTLY in the same set returned ~2,000.
#
# ponytail: an absolute floor, not a comparison against Chromium. It catches a page that
# rendered to nothing; it does NOT catch one that rendered PARTIALLY -- qualcomm.com came
# back with 1,456 characters of Chromium's 7,207 and would pass this check. Closing that
# needs a second opinion per page (render both, compare), which costs the saving this
# class exists for. Raise the floor, or route known-partial hosts straight to Chromium,
# if partial renders start showing up in reports.
MIN_CHARS = int(os.getenv("LIGHTPANDA_MIN_CHARS", "500"))
NAV_TIMEOUT_MS = int(os.getenv("LIGHTPANDA_TIMEOUT_MS", "20000"))


def _dial_url(cdp_url: str) -> str:
    """The CDP URL with its hostname replaced by a resolved address.

    Returns the URL unchanged when it is already an address or cannot be resolved --
    connecting and failing gives a better log line than raising here.
    """
    parsed = urlparse(cdp_url)
    host = parsed.hostname or ""
    if not host or host.replace(".", "").isdigit():
        return cdp_url
    try:
        address = socket.gethostbyname(host)
    except OSError:
        return cdp_url
    port = f":{parsed.port}" if parsed.port else ""
    return urlunparse(parsed._replace(netloc=f"{address}{port}"))


class LightpandaScraper:
    """Lightpanda-first scraper with a Chromium fallback."""

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        self.cdp_url = os.getenv("LIGHTPANDA_CDP_URL", DEFAULT_CDP_URL)
        # Read by the caller after scrape() to explain an empty body (see
        # BeautifulSoupScraper). Set from whichever tier actually answered.
        self.unreadable_reason = None

    def scrape(self) -> tuple:
        """``(content, images, title)``. Never raises -- see Crawl4AIScraper.scrape."""
        content, images, title = "", [], ""
        try:
            content, images, title = self._render()
        except Exception as exc:
            logger.warning(f"Lightpanda could not render {self.link} ({exc}); "
                           "falling back to Chromium")
            return self._fallback()

        if len(content) < MIN_CHARS:
            logger.info(f"Lightpanda returned {len(content)} chars for {self.link} "
                        f"(< {MIN_CHARS}); falling back to Chromium")
            fallback = self._fallback()
            # Keep Lightpanda's result only if Chromium did no better: a page that is
            # genuinely short is not a failure, and re-reporting it as empty would drop a
            # source both engines agreed on.
            return fallback if len(fallback[0]) > len(content) else (content, images, title)

        return content, images, title

    def _render(self) -> tuple:
        """Drive Lightpanda over CDP and parse the DOM it produces.

        Parsed with the same helpers as `BeautifulSoupScraper`, which is what keeps the
        image entries in the `{url, score}` shape the consumers index -- returning bare
        URLs here once cost a whole research run (see [gptr][s18]).
        """
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(_dial_url(self.cdp_url))
            try:
                context = browser.contexts[0] if browser.contexts else browser.new_context()
                page = context.new_page()
                try:
                    # domcontentloaded, not networkidle: measured across the sample the
                    # three wait strategies produced byte-identical text, and networkidle
                    # only added latency on pages that poll.
                    page.goto(self.link, timeout=NAV_TIMEOUT_MS,
                              wait_until="domcontentloaded")
                    html = page.content()
                finally:
                    page.close()
            finally:
                browser.close()

        soup = clean_soup(BeautifulSoup(html, "lxml"))
        content = get_text_from_soup(soup)
        self.unreadable_reason = detect_unreadable(content, html, {}, 200)
        return content, get_relevant_images(soup, self.link), extract_title(soup)

    def _fallback(self) -> tuple:
        """Chromium, through the Crawl4AI container."""
        from ..crawl4ai.crawl4ai import Crawl4AIScraper

        scraper = Crawl4AIScraper(self.link, self.session)
        result = scraper.scrape()
        self.unreadable_reason = getattr(scraper, "unreadable_reason", None)
        return result
