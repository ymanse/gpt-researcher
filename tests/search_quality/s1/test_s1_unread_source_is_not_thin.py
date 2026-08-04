"""D1: a page we were BLOCKED from reading must not be recorded as a page that says little.

Measured 2026-08-04. preprints.org answers this scraper with an Akamai Bot Manager
interstitial: HTTP **200**, 2,670 bytes of well-formed markup, and 32 characters of
extracted text reading "Powered and protected by\\nPrivacy". Nothing raised, no status was
ever inspected (BeautifulSoupScraper never read `response.status_code`), and
`scraper.py`'s `len(content) < 100` guard dropped it under the message "Content too short
or empty for {link}" — a sentence that describes a THIN PAGE.

That single mislabel is the head of a chain. The record is deleted at scraper.py's
`raw_content is not None` filter, so the URL never reaches `research_sources`, never
reaches `_read_docs`, and every consumer downstream reads it through `dict.get(url, "")`.
By the time a verification pass looks, "we were blocked" has become "the source is empty",
which becomes "the source does not support this claim". A correct figure was discarded
that way.

The same 7-URL sample turned up a SECOND silent failure with a different cause and the
identical symptom: news.ycombinator.com answered HTTP 429 with a 6-byte body.

WHAT IS PINNED HERE — outcomes, not mechanism:
  (a) an interstitial is classified as unread, and says why
  (b) a real page is NOT, however much markup surrounds its text
  (c) a genuinely thin page is NOT — the distinction has to survive both ways
  (d) the scraper reports the reason on the dropped record
  (e) raw_content stays None, never "" (an empty string registers as "we hold this page
      and it is blank", which disarms tree_research's evidence-starvation guard and
      suppresses CitationAgent's re-fetch)
  (f) a run that loses pages says so

NO RATIO THRESHOLD. Measured the same day: blocked preprints.org scores 1.20% text/HTML,
GitHub scores 0.85% and a genuinely thin page scores 25.40%. Every ratio threshold flags a
good page and passes a stub, so the detector uses structural signals instead. A test that
re-derived a ratio would be re-introducing the dead hypothesis.
"""
from __future__ import annotations

import logging
from unittest import mock

from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.scraper.utils import WALL_MAX_CHARS, detect_unreadable

# Verbatim shape of the interstitial preprints.org served (trimmed, markers intact).
AKAMAI_HTML = (
    '<!DOCTYPE html><html><head> <meta charset="utf-8"> '
    '<meta http-equiv="refresh" content="5; URL=\'/manuscript/202606.1312/v1'
    '?bm-verify=AAQAAAAN_____zjRPY\'" /><title>&nbsp;</title></head><body>'
    '<div><p>Powered and protected by </p></div>'
    '<img id="akam-logo" src="/_sec/akamai-logo.svg" alt="Powered by Akamai" />'
    '<div class="akamai-privacy"><a href="https://www.akamai.com/privacy">Privacy</a>'
    "</div></body></html>"
)
AKAMAI_TEXT = "Powered and protected by\nPrivacy"

REAL_HTML = "<html><body><article>" + ("<p>Postgres can get you pretty far. </p>" * 60) + "</article></body></html>"
REAL_TEXT = "Postgres can get you pretty far. " * 60

THIN_HTML = (
    "<!doctype html><html><head><title>Example Domain</title></head><body>"
    "<div><h1>Example Domain</h1><p>This domain is for use in illustrative examples in "
    "documents. You may use this domain in literature without prior coordination or "
    "asking for permission.</p></div></body></html>"
)
THIN_TEXT = (
    "Example Domain\nExample Domain\nThis domain is for use in illustrative examples in "
    "documents. You may use this domain in literature without prior coordination or "
    "asking for permission."
)


def test_an_interstitial_served_as_200_is_reported_unread_with_a_reason():
    """(a) The whole defect in one line: status 200, real markup, no page."""
    reason = detect_unreadable(AKAMAI_TEXT, AKAMAI_HTML, {"set-cookie": "ak_bmsc=F9AE"}, 200)
    assert reason, "an Akamai challenge body must not pass as the page"
    assert "bot-challenge" in reason or "meta-refresh" in reason, (
        f"the reason has to name what was seen, not just that something was wrong: {reason}"
    )


def test_a_rate_limited_response_is_reported_unread():
    """The second silent failure in the same sample: HTTP 429, 6-byte body."""
    assert detect_unreadable("", "", None, 429) == "http-429"


def test_a_real_page_is_never_called_unread_however_much_markup_surrounds_it():
    """(b) The direction that kills a naive fix. GitHub's real article scores a LOWER
    text/HTML ratio than the bot wall does."""
    assert detect_unreadable(REAL_TEXT, REAL_HTML, {"set-cookie": "ak_bmsc=X"}, 200) is None, (
        "a bot-management cookie rides on successful responses too — it may not, alone, "
        "condemn a page we actually read"
    )


def test_a_genuinely_thin_page_is_not_called_unread():
    """(c) example.com really is short. Short is not blocked."""
    assert detect_unreadable(THIN_TEXT, THIN_HTML, {}, 200) is None


CHATTY_WALL_TEXT = (
    "Checking your browser before accessing this site. This process is automatic. "
    "Your browser will redirect to your requested content shortly. Please allow up to "
    "five seconds. DDoS protection by Cloudflare. Ray ID: 8f2c1a9b4e7d0000."
)
CHATTY_WALL_HTML = f"<html><body><div id='cf-wrapper'>{CHATTY_WALL_TEXT}</div></body></html>"


def test_a_challenge_that_is_not_short_is_still_caught():
    """A bot wall is under no obligation to be brief.

    The strong signals exist precisely so length cannot be the gate: this banner runs
    well past the 100-character floor, so anything that consulted the detector only for
    short pages would file it as article text — the exact failure the change is for.
    """
    assert len(CHATTY_WALL_TEXT) > 100, "the point of this fixture is that it is long"
    assert detect_unreadable(CHATTY_WALL_TEXT, CHATTY_WALL_HTML, {}, 200), (
        "a verbose interstitial must not pass merely by being verbose"
    )


def test_an_article_about_access_denied_errors_is_not_mistaken_for_one():
    """The false positive that the first cut of this fix actually shipped.

    Making the banner wording a strong, length-independent signal condemned AWS's
    "Troubleshoot access denied (403 Forbidden) errors in Amazon S3" — 16,009 characters
    of genuine documentation, measured live 2026-08-04, verdict
    `bot-challenge:access denied`. That is the worst possible failure for a research
    tool: it deletes precisely the page that explains the error being researched.

    So the wording is trusted only on a document short enough to BE the banner, while
    the machine tokens (bm-verify, cdn-cgi/challenge-platform) stay length-independent
    because no article can emit those.
    """
    article = "Troubleshoot access denied (403 Forbidden) errors in Amazon S3. " * 260
    html = f"<html><title>Troubleshoot access denied errors</title><body>{article}</body></html>"

    assert len(article) > WALL_MAX_CHARS
    assert detect_unreadable(article, html, {}, 200) is None, (
        "an article that discusses access-denied errors is not an access-denied page"
    )


def test_a_machine_token_is_trusted_at_any_length():
    """The other side of the split: challenge machinery cannot appear in prose."""
    long_body = "Some ordinary article text that goes on for a while. " * 40
    html = f"<html><script>/cdn-cgi/challenge-platform/h/b/orchestrate</script>{long_body}</html>"

    assert len(long_body) > WALL_MAX_CHARS
    assert detect_unreadable(long_body, html, {}, 200), (
        "no article emits a cdn-cgi challenge-platform path; length must not excuse it"
    )


def test_an_error_page_with_plenty_of_text_is_still_not_the_document():
    """A 404 body can be chatty too. An error response is never what we asked for."""
    friendly_404 = (
        "Sorry, we could not find the page you were looking for. It may have been "
        "moved or deleted. Try our search, or head back to the homepage for more."
    )
    assert len(friendly_404) > 100
    assert detect_unreadable(friendly_404, f"<html>{friendly_404}</html>", {}, 404) == "http-404"


class _FakeScraper:
    """Stands in for BeautifulSoupScraper: returns the interstitial's text and reports why."""

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        self.unreadable_reason = None

    def scrape(self):
        self.unreadable_reason = detect_unreadable(
            AKAMAI_TEXT, AKAMAI_HTML, {"set-cookie": "ak_bmsc=F9AE"}, 200
        )
        return AKAMAI_TEXT, [], "preprints"


async def _run_one(caplog):
    scraper = Scraper(
        ["https://www.preprints.org/manuscript/202606.1312/v1"],
        "pytest-agent",
        "bs",
        worker_pool=mock.MagicMock(executor=None),
    )
    # get_scraper is the seam, not SCRAPER_CLASSES: that dict is a LOCAL inside
    # get_scraper, so there is nothing at module scope to patch.
    with mock.patch.object(Scraper, "get_scraper", return_value=_FakeScraper), \
            caplog.at_level(logging.WARNING):
        kept = await scraper.run()
    return scraper, kept


async def test_the_dropped_record_carries_the_reason_and_never_an_empty_document(caplog):
    """(d) + (e). The record must say WHY, and must not claim we hold the page."""
    scraper, kept = await _run_one(caplog)

    assert kept == [], "a page we could not read must not be handed on as a source"
    assert len(scraper.unread) == 1, f"the loss has to be recorded somewhere: {scraper.unread}"
    entry = scraper.unread[0]
    assert entry["reason"] != "content-too-short", (
        "'too short' is the mislabel this test exists to prevent — the reason must name "
        f"the challenge: {entry}"
    )
    assert "bot-challenge" in entry["reason"] or "meta-refresh" in entry["reason"]


class _ChattyWallScraper(_FakeScraper):
    """A wall that clears the 100-char floor — the case a length-gated fix misses."""

    def scrape(self):
        self.unreadable_reason = detect_unreadable(CHATTY_WALL_TEXT, CHATTY_WALL_HTML, {}, 200)
        return CHATTY_WALL_TEXT, [], "Just a moment..."


async def test_a_long_challenge_page_never_becomes_research_content(caplog):
    """End to end, not just in the detector.

    Detecting a wall is worth nothing if the pipeline only asks about pages it already
    considers too short. This pins the consumption side: the banner is long, so the
    length guard would pass it, and it must still be refused.
    """
    scraper = Scraper(
        ["https://walled.example.com/article"],
        "pytest-agent",
        "bs",
        worker_pool=mock.MagicMock(executor=None),
    )
    with mock.patch.object(Scraper, "get_scraper", return_value=_ChattyWallScraper), \
            caplog.at_level(logging.WARNING):
        kept = await scraper.run()

    assert kept == [], (
        "a 200-character challenge banner cleared the length floor and would have "
        "entered the corpus as the article"
    )
    assert scraper.unread[0]["reason"] != "content-too-short"


async def test_a_run_that_loses_pages_says_so(caplog):
    """(f) Nineteen of twenty URLs could vanish and no number in the run moved."""
    _, _ = await _run_one(caplog)
    warnings = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "were NOT read" in warnings, (
        f"the end of a run must state how many sources it never obtained: {warnings!r}"
    )
    assert "1 of 1" in warnings
