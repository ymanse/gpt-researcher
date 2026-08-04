"""Utility functions for web scraping.

This module provides helper functions for extracting content, images,
and processing HTML from web pages.
"""

import hashlib
import logging
import re
from urllib.parse import parse_qs, urljoin, urlparse

import bs4
from bs4 import BeautifulSoup


# Cookies the bot-management vendors set themselves. Header-level, so they do not depend
# on the challenge page's wording or language.
CHALLENGE_COOKIES = ("ak_bmsc", "__cf_bm", "cf_clearance", "datadome", "incap_ses",
                     "visid_incap", "_px", "px-captcha")
# THE MACHINERY the walls ship. These tokens exist nowhere but inside the challenge
# itself — bm-verify and triggerInterstitialChallenge are Akamai's (measured on
# preprints.org 2026-08-04), the cdn-cgi path is Cloudflare's — so they are safe to
# believe on a document of any length.
CHALLENGE_TOKENS = re.compile(
    r"bm-verify|triggerinterstitialchallenge|/cdn-cgi/challenge-platform|"
    r"__cf_chl_|px-captcha",
    re.I,
)
# WHAT THE WALL SAYS. Trustworthy only on a document short enough to be nothing BUT the
# banner. These are ordinary English and real articles are written about them: AWS's
# "Troubleshoot access denied (403 Forbidden) errors in Amazon S3" is 16,009 characters
# of genuine documentation and matches `access denied` (measured 2026-08-04). Believing
# a banner on a full-length page deletes exactly the sources a research run wants most —
# the ones that explain the error you are researching.
CHALLENGE_BANNERS = re.compile(
    r"powered and protected by|just a moment|checking your browser|"
    r"enable javascript and cookies to continue|attention required!|"
    r"verify(?:ing)? you are (?:a )?human|access denied|request unsuccessful",
    re.I,
)
# Longest wall measured is 275 characters; the shortest real article in the sample is
# 2,119. The bound sits in that gap, nearer the walls.
WALL_MAX_CHARS = 1000
# A refresh redirect means the response is a WAYPOINT, not the document. A real article
# does not ask the browser to leave five seconds after arriving.
META_REFRESH = re.compile(r"<meta[^>]+http-equiv=[\"']?refresh[\"']?", re.I)
MIN_USABLE_CHARS = 100  # the bar scraper.py already drops below


def detect_unreadable(text: str, html: str, headers=None, status: int = 200):
    """Why this response is not the page — or None if it looks like the page.

    A fetch can fail while returning HTTP 200 and a well-formed document. Measured
    2026-08-04, preprints.org answers our scraper with an Akamai interstitial: 200 OK,
    2,670 bytes of markup, 32 characters of text reading "Powered and protected by
    Privacy". Nothing raised, so the page was recorded as one that had almost nothing to
    say — and a later verification pass, comparing a correct figure against that
    non-corpus, reported the SOURCE as disagreeing. Absence became evidence.

    Returns a REASON rather than a bool so the log and the run summary can say which
    signal fired.

    STRONG signals fire unconditionally: the challenge markers and the refresh waypoint
    appear only on an interstitial, so seeing them means the response is not the
    document however much boilerplate it carries.

    WEAK signals fire only once the extracted text is already below what the pipeline
    will use. A vendor cookie is the clearest reason for that split: ak_bmsc is set by
    Akamai Bot Manager on EVERY response from a protected origin, successful ones
    included, so alone it says "bot management is present", not "you were blocked" —
    treating it as proof misclassified four of seven good pages when this was derived.

    Below that bar the question stops being "was this a bot wall" and becomes "did we
    fail to read a document that has content" — and blocked, rate-limited, JS-rendered
    and paywalled all take the same remedy: do not report the source as read.

    Do NOT replace this with a text/HTML ratio. That was tried and measured dead on the
    same day: blocked preprints.org scores 1.20% while GitHub scores 0.85% and a
    genuinely thin page (example.com) scores 25.40%, so every ratio threshold flags a
    good page and passes a stub.
    """
    head = (html or "")[:4000]
    body = (text or "").strip()

    # STRONG — machine tokens and the status line. Neither can be produced by an article
    # writing ABOUT bot walls, so length is irrelevant to them.
    m = CHALLENGE_TOKENS.search(head)
    if m:
        return f"bot-challenge:{m.group(0)[:40].strip().lower()}"
    if status >= 400:
        # An error body is never the document we asked for, however chatty it is: a 404
        # page or a rate-limit notice can easily run past MIN_USABLE_CHARS, and
        # accepting it files a fetch that failed as a source that spoke.
        return f"http-{status}"

    # BANNER-SHAPED — the wording only means anything on a document that is nothing but
    # the wording. Above this size the same phrases are simply the subject matter.
    if len(body) < WALL_MAX_CHARS:
        m = CHALLENGE_BANNERS.search(head)
        if m:
            return f"bot-challenge:{m.group(0)[:40].strip().lower()}"
        if META_REFRESH.search(head):
            # On a stub this is the challenge's own waypoint; on a full page it is an
            # ordinary canonical redirect, and condemning that would lose real sources.
            return "meta-refresh-redirect"

    if len(body) >= MIN_USABLE_CHARS:
        return None  # we got the page; nothing below outweighs that

    cookie = ""
    try:
        cookie = str(headers.get("set-cookie", "")) if headers else ""
    except AttributeError:  # a mapping-like that is not a mapping — not worth crashing on
        cookie = ""
    for c in CHALLENGE_COOKIES:
        if c.lower() in cookie.lower():
            return f"bot-managed-origin:{c}"
    if len(html or "") > 1500:
        # Substantial markup, nothing legible: JS-rendered or paywalled.
        return "no-text-in-markup"
    if len(html or "") < 200:
        # Not a thin page — a response with no document in it at all.
        return "empty-response"
    return None


def get_relevant_images(soup: BeautifulSoup, url: str) -> list:
    """Extract relevant images from the page"""
    image_urls = []
    
    try:
        # Find all img tags with src attribute
        all_images = soup.find_all('img', src=True)
        
        for img in all_images:
            img_src = urljoin(url, img['src'])
            if img_src.startswith(('http://', 'https://')):
                score = 0
                # Check for relevant classes
                if any(cls in img.get('class', []) for cls in ['header', 'featured', 'hero', 'thumbnail', 'main', 'content']):
                    score = 4  # Higher score
                # Check for size attributes
                elif img.get('width') and img.get('height'):
                    width = parse_dimension(img['width'])
                    height = parse_dimension(img['height'])
                    if width and height:
                        if width >= 2000 and height >= 1000:
                            score = 3  # Medium score (very large images)
                        elif width >= 1600 or height >= 800:
                            score = 2  # Lower score
                        elif width >= 800 or height >= 500:
                            score = 1  # Lowest score
                        elif width >= 500 or height >= 300:
                            score = 0  # Lowest score
                        else:
                            continue  # Skip small images
                
                image_urls.append({'url': img_src, 'score': score})
        
        # Sort images by score (highest first)
        sorted_images = sorted(image_urls, key=lambda x: x['score'], reverse=True)
        
        return sorted_images[:10]  # Ensure we don't return more than 10 images in total
    
    except Exception as e:
        logging.error(f"Error in get_relevant_images: {e}")
        return []

def parse_dimension(value: str) -> int:
    """Parse dimension value, handling px units"""
    if value.lower().endswith('px'):
        value = value[:-2]  # Remove 'px' suffix
    try:
        # Convert to float first to handle decimal values like '409.12'
        return int(float(value))
    except (ValueError, TypeError) as e:
        print(f"Error parsing dimension value {value}: {e}")
        return None

def extract_title(soup: BeautifulSoup) -> str:
    """Extract the title from the BeautifulSoup object"""
    return soup.title.string if soup.title else ""

def get_image_hash(image_url: str) -> str:
    """Calculate a simple hash based on the image filename and essential query parameters"""
    try:
        parsed_url = urlparse(image_url)
        
        # Extract the filename
        filename = parsed_url.path.split('/')[-1]
        
        # Extract essential query parameters (e.g., 'url' for CDN-served images)
        query_params = parse_qs(parsed_url.query)
        essential_params = query_params.get('url', [])
        
        # Combine filename and essential parameters
        image_identifier = filename + ''.join(essential_params)
        
        # Calculate hash
        return hashlib.md5(image_identifier.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating image hash for {image_url}: {e}")
        return None


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean the soup by removing unwanted tags"""
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "header",
            "nav",
            "menu",
            "sidebar",
            "svg",
        ]
    ):
        tag.decompose()

    disallowed_class_set = {"nav", "menu", "sidebar", "footer"}

    # clean tags with certain classes
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False

        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()

    return soup


def get_text_from_soup(soup: BeautifulSoup) -> str:
    """Get the relevant text from the soup with improved filtering"""
    text = soup.get_text(strip=True, separator="\n")
    # Remove excess whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text