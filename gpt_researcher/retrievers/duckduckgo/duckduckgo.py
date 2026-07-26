from ..utils import check_pkg, normalize_wikipedia_lang

# ddgs "worldwide" default; its lang segment "wt" is a placeholder, not a language.
_DEFAULT_REGION = 'wt-wt'


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query, query_domains=None):
        check_pkg('ddgs')
        from ddgs import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or None

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        # ddgs fans the region out to every engine; its wikipedia engine builds
        # https://{lang}.wikipedia.org from the lang segment, so 'wt-wt' hit
        # wt.wikipedia.org (DNS fail). Keep the country segment, fix the lang.
        country, _, lang = _DEFAULT_REGION.partition('-')
        region = f"{country}-{normalize_wikipedia_lang(lang)}"
        # TODO: Add support for query domains
        try:
            search_response = self.ddg.text(self.query, region=region, max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
