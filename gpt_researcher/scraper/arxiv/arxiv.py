import re

import arxiv


class ArxivScraper:
    """Metadata + abstract for an arxiv.org/abs/ landing page.

    Was built on langchain_community's ArxivRetriever, which calls
    `arxiv.Search(...).results()` — removed in arxiv 2.0 and long gone from the 4.0.0 the
    container ships. Every arxiv.org URL therefore died with
    "'Search' object has no attribute 'results'" and was dropped from the research context
    without ever surfacing as a missing source. Measured on one run: 4 of 4 arxiv URLs
    lost, on a query whose best sources were all arxiv papers.

    Full text is NOT fetched here: arxiv.org/html/ and /pdf/ links are routed to the
    ordinary HTML and PDF scrapers (see Scraper.get_scraper), which read the whole paper.
    This path exists for /abs/, where the API answer is cleaner than scraping the landing
    page.
    """

    # 2509.08304 / 2509.08304v2 / hep-th/9901001 — tolerant of a trailing .pdf, query
    # string or fragment, which search results routinely append.
    _ID = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def paper_id(self) -> str:
        m = self._ID.search(self.link)
        return m.group(1) if m else ""

    def scrape(self):
        """Returns (context, images, title), or empty values when the paper cannot be
        resolved — the caller reads an empty context as "nothing scraped", which is the
        honest outcome; raising here would cost the rest of the batch."""
        pid = self.paper_id()
        if not pid:
            return "", [], ""
        result = next(arxiv.Client().results(arxiv.Search(id_list=[pid])), None)
        if result is None:
            return "", [], ""

        # Published date and authors stay in the context so the report can cite in APA
        # style, exactly as the previous implementation did.
        authors = ", ".join(a.name for a in result.authors)
        context = (f"Published: {result.published}; Author: {authors}; "
                   f"Content: {result.summary}")
        return context, [], result.title
