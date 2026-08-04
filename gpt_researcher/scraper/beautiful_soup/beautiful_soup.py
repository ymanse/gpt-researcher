from bs4 import BeautifulSoup

from ..utils import (
    detect_unreadable,
    get_relevant_images,
    extract_title,
    get_text_from_soup,
    clean_soup,
)

class BeautifulSoupScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        # Why this fetch produced nothing usable, when it did — read by Scraper.run()
        # after scrape() returns. An ATTRIBUTE rather than a fourth return value: every
        # scraper in this package returns (content, images, title) and callers unpack
        # positionally, so widening the tuple would break all of them for a field only
        # one scraper can currently fill.
        self.unreadable_reason = None

    def scrape(self):
        """
        This function scrapes content from a webpage by making a GET request, parsing the HTML using
        BeautifulSoup, and extracting script and style elements before returning the cleaned content.
        
        Returns:
          The `scrape` method is returning the cleaned and extracted content from the webpage specified
        by the `self.link` attribute. The method fetches the webpage content, removes script and style
        tags, extracts the text content, and returns the cleaned content as a string. If any exception
        occurs during the process, an error message is printed and an empty string is returned.
        """
        try:
            response = self.session.get(self.link, timeout=4)
            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=response.encoding
            )

            soup = clean_soup(soup)

            content = get_text_from_soup(soup)

            image_urls = get_relevant_images(soup, self.link)

            # Extract the title using the utility function
            title = extract_title(soup)

            # WAS THIS THE PAGE, OR A WALL? The status code was never read here and a
            # challenge body was parsed as if it were the article: preprints.org answers
            # 200 with an Akamai interstitial, which reached the pipeline as a page that
            # simply had little to say. Recorded, not raised — the caller decides.
            # response.text, not a hand-rolled decode: bytes.decode raises LookupError
            # for a charset label Python does not know (errors="replace" does NOT cover
            # an unknown CODEC, only undecodable bytes), and a scraper must not die on a
            # site that mislabels its encoding. requests already resolves this.
            self.unreadable_reason = detect_unreadable(
                content, response.text, response.headers, response.status_code
            )

            return content, image_urls, title

        except Exception as e:
            print("Error! : " + str(e))
            self.unreadable_reason = f"{type(e).__name__}: {str(e)[:120]}"
            return "", [], ""