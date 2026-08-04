"""Web scraper module for GPT Researcher.

This module provides the Scraper class that extracts content from URLs
using various scraping backends (BeautifulSoup, PyMuPDF, Browser, etc.).
"""

import asyncio
import importlib
import logging
import subprocess
import sys

import requests
from colorama import Fore, init

from gpt_researcher.utils.workers import WorkerPool

from . import (
    ArxivScraper,
    BeautifulSoupScraper,
    BrowserScraper,
    FireCrawl,
    NoDriverScraper,
    PyMuPDFScraper,
    TavilyExtract,
    WebBaseLoaderScraper,
)


class Scraper:
    """
    Scraper class to extract the content from the links
    """

    def __init__(self, urls, user_agent, scraper, worker_pool: WorkerPool):
        """
        Initialize the Scraper class.
        Args:
            urls: List of URLs to scrape (duplicates will be removed)
        """
        # Optimization: Remove duplicate URLs to avoid redundant scraping
        unique_urls = list(dict.fromkeys(urls))  # Preserves order while removing duplicates
        duplicates_removed = len(urls) - len(unique_urls)

        self.urls = unique_urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.scraper = scraper
        if self.scraper == "tavily_extract":
            self._check_pkg(self.scraper)
        if self.scraper == "firecrawl":
            self._check_pkg(self.scraper)
        self.logger = logging.getLogger(__name__)
        self.worker_pool = worker_pool

        # Log deduplication results if duplicates were found
        if duplicates_removed > 0:
            self.logger.info(
                f"Removed {duplicates_removed} duplicate URL(s). "
                f"Scraping {len(unique_urls)} unique URLs instead of {len(urls)}."
            )

    async def run(self):
        """
        Extracts the content from the links
        """
        contents = await asyncio.gather(
            *(self.extract_data_from_url(url, self.session) for url in self.urls)
        )

        res = [content for content in contents if content["raw_content"] is not None]

        # COUNT WHAT WAS LOST. Dropping the record here is right — an entry with no text
        # must not masquerade as a page we hold — but until now the drop left no trace
        # of any kind: no counter, no ratio, no end-of-run line. Nineteen of twenty URLs
        # could vanish and every number in the run stayed the same, so a report built on
        # one source looked exactly like a report built on twenty. The tally is kept on
        # the instance as well as logged, so a caller that wants it does not have to
        # scrape its own logs.
        self.unread = [
            {"url": c["url"], "reason": c.get("unread_reason") or "unknown"}
            for c in contents
            if c["raw_content"] is None
        ]
        if self.unread:
            by_reason: dict = {}
            for u in self.unread:
                key = u["reason"].split(":")[0]
                by_reason[key] = by_reason.get(key, 0) + 1
            self.logger.warning(
                f"{len(self.unread)} of {len(contents)} URLs were NOT read "
                f"({', '.join(f'{k}={v}' for k, v in sorted(by_reason.items()))}). "
                f"Claims resting on them cannot be verified from the text — that is a "
                f"gap in the evidence, not evidence against the claim."
            )
        return res

    def _check_pkg(self, scrapper_name: str) -> None:
        """
        Checks and ensures required Python packages are available for scrapers that need
        dependencies beyond requirements.txt. When adding a new scraper to the repo, update `pkg_map`
        with its required information and call check_pkg() during initialization.
        """
        pkg_map = {
            "tavily_extract": {
                "package_installation_name": "tavily-python",
                "import_name": "tavily",
            },
            "firecrawl": {
                "package_installation_name": "firecrawl-py",
                "import_name": "firecrawl",
            },
        }
        pkg = pkg_map[scrapper_name]
        if not importlib.util.find_spec(pkg["import_name"]):
            pkg_inst_name = pkg["package_installation_name"]
            init(autoreset=True)
            print(Fore.YELLOW + f"{pkg_inst_name} not found. Attempting to install...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_inst_name]
                )
                importlib.invalidate_caches()
                print(Fore.GREEN + f"{pkg_inst_name} installed successfully.")
            except subprocess.CalledProcessError:
                raise ImportError(
                    Fore.RED
                    + f"Unable to install {pkg_inst_name}. Please install manually with "
                    f"`pip install -U {pkg_inst_name}`"
                )

    async def extract_data_from_url(self, link, session):
        """
        Extracts the data from the link with logging
        """
        async with self.worker_pool.throttle():
            try:
                Scraper = self.get_scraper(link)
                scraper = Scraper(link, session)

                # Get scraper name
                scraper_name = scraper.__class__.__name__
                self.logger.info(f"\n=== Using {scraper_name} ===")

                # Get content
                if hasattr(scraper, "scrape_async"):
                    content, image_urls, title = await scraper.scrape_async()
                else:
                    (
                        content,
                        image_urls,
                        title,
                    ) = await asyncio.get_running_loop().run_in_executor(
                        self.worker_pool.executor, scraper.scrape
                    )

                # The reason is consulted BEFORE the length test, not inside it. A
                # challenge page is not obliged to be short — "Checking your browser…
                # This process is automatic, you will be redirected shortly" clears 100
                # characters easily — and gating the strong signals behind the length
                # test would have let exactly those through as article text, which is
                # the failure this whole change exists to stop.
                reason = getattr(scraper, "unreadable_reason", None)
                if reason or len(content) < 100:
                    # SAY WHICH IT WAS. "Content too short or empty" describes a thin
                    # page, and a bot wall, a 429 and a JS-only shell all arrived here
                    # wearing that description — so a source we were BLOCKED from
                    # reading was recorded as a source with nothing to say, and a later
                    # verification pass read that silence as the source disagreeing.
                    # Measured 2026-08-04 on preprints.org (Akamai, HTTP 200, 32 chars)
                    # and news.ycombinator.com (HTTP 429, 6 bytes).
                    if reason:
                        self.logger.warning(
                            f"UNREAD {link}: {reason} "
                            f"(extracted {len(content)} chars) — the page was not "
                            f"obtained; do not treat it as a source that says little"
                        )
                    else:
                        self.logger.warning(f"Content too short or empty for {link}")
                    return {
                        "url": link,
                        # stays None on purpose: an empty string here would register as
                        # "we hold this page and it is blank", which disarms the
                        # evidence-starvation guard in tree_research and suppresses
                        # CitationAgent's re-fetch (see deep_research.scraped_documents)
                        "raw_content": None,
                        "unread_reason": reason or "content-too-short",
                        "image_urls": [],
                        "title": title,
                    }

                # Log results
                self.logger.info(f"\nTitle: {title}")
                self.logger.info(
                    f"Content length: {len(content) if content else 0} characters"
                )
                self.logger.info(f"Number of images: {len(image_urls)}")
                self.logger.info(f"URL: {link}")
                self.logger.info("=" * 50)

                # (the duplicate `len(content) < 100` guard that stood here was
                # unreachable — the branch above returns for every such case)
                return {
                    "url": link,
                    "raw_content": content,
                    "image_urls": image_urls,
                    "title": title,
                }

            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                return {"url": link, "raw_content": None, "image_urls": [], "title": ""}

    def get_scraper(self, link):
        """
        The function `get_scraper` determines the appropriate scraper class based on the provided link
        or a default scraper if none matches.

        Args:
          link: The `get_scraper` method takes a `link` parameter which is a URL link to a webpage or a
        PDF file. Based on the type of content the link points to, the method determines the appropriate
        scraper class to use for extracting data from that content.

        Returns:
          The `get_scraper` method returns the scraper class based on the provided link. The method
        checks the link to determine the appropriate scraper class to use based on predefined mappings
        in the `SCRAPER_CLASSES` dictionary. If the link ends with ".pdf", it selects the
        `PyMuPDFScraper` class. If the link contains "arxiv.org", it selects the `ArxivScraper
        """

        SCRAPER_CLASSES = {
            "pdf": PyMuPDFScraper,
            "arxiv": ArxivScraper,
            "bs": BeautifulSoupScraper,
            "web_base_loader": WebBaseLoaderScraper,
            "browser": BrowserScraper,
            "nodriver": NoDriverScraper,
            "tavily_extract": TavilyExtract,
            "firecrawl": FireCrawl,
        }

        scraper_key = None

        # arxiv is split by URL SHAPE, not by host. Routing every arxiv.org link to
        # ArxivScraper sent full-text pages through an abstract-only path: measured on one
        # research run, 3 of the 4 arxiv URLs the retrievers found were arxiv.org/html/,
        # i.e. the whole paper as ordinary HTML that the default scraper reads fine.
        # /pdf/ links carry no .pdf suffix, so they need naming here or BeautifulSoup is
        # handed a PDF binary.
        if link.endswith(".pdf") or "arxiv.org/pdf/" in link:
            scraper_key = "pdf"
        elif "arxiv.org/abs/" in link:
            scraper_key = "arxiv"
        else:
            scraper_key = self.scraper

        scraper_class = SCRAPER_CLASSES.get(scraper_key)
        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class
