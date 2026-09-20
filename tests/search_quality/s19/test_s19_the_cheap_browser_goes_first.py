"""Lightpanda answers first, and Chromium catches what it drops.

Lightpanda runs V8 and the DOM but implements no layout, paint or compositing, which is
where its ~5MB-against-~430MB and ~3x speed come from. The same omission is why it
returns nothing on pages that need a rendering surface. Measured on this host over 24
URLs from real research runs, both engines rendering and the SAME extractor parsing:
21 within 20% of Chromium, three not (stackoverflow 2170 -> 41 chars,
materializedview 15842 -> 170, qualcomm 7207 -> 1456).

So the router is the product, not the browser: cheap engine first, expensive engine for
what comes back thin. What these tests pin is that routing, because every way it can go
wrong is quiet -- a thin page that is never re-fetched reaches the report as a source
that simply had little to say.

They run offline. Playwright is import-mocked; the live path is exercised against the
containers by hand.
"""
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from gpt_researcher.scraper.lightpanda import lightpanda as lp
from gpt_researcher.scraper.lightpanda.lightpanda import LightpandaScraper, _dial_url

ARTICLE = "<html><head><title>Real Page</title></head><body>" + ("word " * 400) + "</body></html>"
SHELL = "<html><head><title>Shell</title></head><body><div id=root></div></body></html>"


def _playwright(html, boom=None):
    """A `playwright.sync_api` module whose browser returns `html`."""
    page = mock.MagicMock()
    if boom:
        page.goto.side_effect = boom
    page.content.return_value = html
    context = mock.MagicMock(); context.new_page.return_value = page
    browser = mock.MagicMock(); browser.contexts = [context]
    chromium = mock.MagicMock(); chromium.connect_over_cdp.return_value = browser
    pw = mock.MagicMock(); pw.chromium = chromium
    entered = mock.MagicMock(); entered.__enter__.return_value = pw
    return SimpleNamespace(sync_playwright=mock.Mock(return_value=entered)), chromium


def _scrape(html, boom=None, fallback=("CHROMIUM BODY " * 50, [], "From Chromium")):
    module, chromium = _playwright(html, boom)
    with mock.patch.dict(sys.modules, {"playwright": mock.MagicMock(),
                                       "playwright.sync_api": module}), \
         mock.patch("gpt_researcher.scraper.crawl4ai.crawl4ai.Crawl4AIScraper") as c4:
        c4.return_value.scrape.return_value = fallback
        c4.return_value.unreadable_reason = None
        result = LightpandaScraper("https://example.com/a").scrape()
    return result, c4, chromium


# ---------------------------------------------------------------- the cheap path

def test_a_rendered_page_never_reaches_chromium():
    """The saving only exists if the common case stops at the first tier."""
    (content, images, title), c4, _ = _scrape(ARTICLE)

    assert title == "Real Page" and len(content) > 500, f"got {title!r}, {len(content)} chars"
    assert not c4.called, "Chromium was asked for a page Lightpanda had already rendered"


def test_images_keep_the_shape_the_consumers_index():
    """Parsed with the same helpers as BeautifulSoupScraper, so entries stay
    `{url, score}`. Returning bare URLs from a scraper cost a whole research run once --
    every scrape raised inside the loop and the node reported 0 context chars
    (see [gptr][s18])."""
    html = ARTICLE.replace("<body>", '<body><img src="/a.png" width="900" height="600">')
    (_, images, _), _, _ = _scrape(html)

    assert images, "no images parsed"
    assert all(isinstance(i, dict) and {"url", "score"} <= set(i) for i in images), images
    sorted(images, key=lambda im: im["score"], reverse=True)[0]["url"]   # browser.py:116


# ------------------------------------------------------------------ the fallback

def test_a_shell_page_falls_through_to_chromium():
    """The failure this router exists for: HTTP 200, a title, and no article -- which is
    exactly what a hydration-dependent SPA gives an engine with no rendering surface."""
    (content, _, title), c4, _ = _scrape(SHELL)

    assert c4.called, f"a {len(content)}-char shell was accepted without asking Chromium"
    assert "CHROMIUM" in content and title == "From Chromium", f"got {title!r}"


def test_a_render_error_falls_through_instead_of_raising():
    """One raising scraper loses every other URL in its batch --
    `Scraper.extract_data_from_url` has no per-URL recovery."""
    (content, _, _), c4, _ = _scrape(ARTICLE, boom=RuntimeError("Target closed"))

    assert c4.called and "CHROMIUM" in content, "a render error did not degrade"


def test_playwright_missing_still_degrades():
    """The library is installed in the image, not in every environment that imports this
    package -- the harness drives the scrapers directly."""
    with mock.patch.dict(sys.modules, {"playwright": None, "playwright.sync_api": None}), \
         mock.patch("gpt_researcher.scraper.crawl4ai.crawl4ai.Crawl4AIScraper") as c4:
        c4.return_value.scrape.return_value = ("CHROMIUM", [], "t")
        c4.return_value.unreadable_reason = None
        content, _, _ = LightpandaScraper("https://example.com/a").scrape()

    assert content == "CHROMIUM", f"got {content!r} with playwright unimportable"


def test_a_genuinely_short_page_keeps_lightpandas_answer():
    """A short page is not a failed render. If Chromium does no better, re-reporting the
    fallback would drop a source both engines agreed on -- and would hide that the page
    really is that short."""
    short = "<html><head><title>Tiny</title></head><body>a short note</body></html>"
    (content, _, title), c4, _ = _scrape(short, fallback=("", [], ""))

    assert c4.called, "the thin page should still have been checked against Chromium"
    assert "short note" in content and title == "Tiny", f"got {title!r}: {content!r}"


# --------------------------------------------------------------- the host problem

@pytest.mark.parametrize("given,expect_changed", [
    ("ws://lightpanda:9222/", True),
    ("ws://172.18.0.6:9222/", False),
])
def test_a_service_name_is_dialled_by_address(given, expect_changed, monkeypatch):
    """Lightpanda answers 403 "Host not allowed" to a WebSocket addressed by name --
    DNS-rebinding protection. That is what stopped Crawl4AI's own `cdp_url` from ever
    reaching it, so the name is resolved here and the address dialled, which keeps the
    compose service name in config (container IPs move)."""
    monkeypatch.setattr(lp.socket, "gethostbyname", lambda host: "10.1.2.3")
    dialled = _dial_url(given)

    assert (dialled != given) is expect_changed, f"{given} -> {dialled}"
    if expect_changed:
        assert "10.1.2.3:9222" in dialled, dialled


def test_an_unresolvable_name_is_left_alone(monkeypatch):
    """Failing to resolve is not worth raising over: dialling and failing produces a
    better log line than an exception from a helper."""
    monkeypatch.setattr(lp.socket, "gethostbyname",
                        lambda host: (_ for _ in ()).throw(OSError("no such host")))
    assert _dial_url("ws://nope:9222/") == "ws://nope:9222/"
