import inspect
import os

from ..utils import check_pkg


def _accepted_by(fn, kwargs):
    """Drop the kwargs `fn` does not declare, rather than forwarding them blindly.

    exa-py's parameter set moves between releases. 2.20.0 (in the image since
    2026-09-06) no longer takes `use_autoprompt`, so on 2026-09-09 every exa call died
    with "Exa.search() got an unexpected keyword argument 'use_autoprompt'" - twice in a
    row, which retires exa from SmartRetriever routing for the rest of the process.
    Re-pinning the retriever to 2.20.0's argument list would just move that break to the
    next release, so read the signature off the client we were actually handed.

    A client declaring **kwargs gets everything: `Exa.search_and_contents` is
    `(self, query, **kwargs)` in 2.20.0, and filtering that by name would silently strip
    the search mode, the result limit and the domain filter, leaving unfiltered defaults
    - a quieter failure than the loud one this replaces.
    """
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in params}


class ExaSearch:
    """
    Exa API Retriever
    """

    def __init__(self, query, query_domains=None):
        """
        Initializes the ExaSearch object.
        Args:
            query: The search query.
        """
        # This validation is necessary since exa_py is optional
        check_pkg("exa_py")
        from exa_py import Exa
        self.query = query
        self.api_key = self._retrieve_api_key()
        self.client = Exa(api_key=self.api_key)
        self.query_domains = query_domains or None

    def _retrieve_api_key(self):
        """
        Retrieves the Exa API key from environment variables.
        Returns:
            The API key.
        Raises:
            Exception: If the API key is not found.
        """
        try:
            api_key = os.environ["EXA_API_KEY"]
        except KeyError:
            raise Exception(
                "Exa API key not found. Please set the EXA_API_KEY environment variable. "
                "You can obtain your key from https://exa.ai/"
            )
        return api_key

    def search(self, max_results=10, search_type="neural", **filters):
        """
        Searches the query using the Exa API.
        Args:
            max_results: The maximum number of results to return.
            search_type: The type of search (e.g., "neural", "keyword").
            **filters: Additional filters (e.g., date range, domains).
        Returns:
            A list of search results.
        """
        # `search_type` keeps its name because SmartRetriever routes the code_technical
        # bundle by inspecting this method for exactly that parameter.
        results = self.client.search(self.query, **_accepted_by(self.client.search, {
            "type": search_type,
            "num_results": max_results,
            "include_domains": self.query_domains,
            **filters,
        }))

        # `Result.text` is None unless contents were requested, and every other retriever
        # here returns a string body: SmartRetriever._deduplicate_results calls len() on
        # it, and since the key exists its "" default never fires.
        search_response = [
            {"href": result.url, "body": result.text or ""} for result in results.results
        ]
        return search_response

    def find_similar(self, url, exclude_source_domain=False, **filters):
        """
        Finds similar documents to the provided URL using the Exa API.
        Args:
            url: The URL to find similar documents for.
            exclude_source_domain: Whether to exclude the source domain in the results.
            **filters: Additional filters.
        Returns:
            A list of similar documents.
        """
        # 2.20.0 still declares exclude_source_domain, but the blind **filters forwarding
        # is the same defect that killed search(), so it goes through the same gate.
        results = self.client.find_similar(url, **_accepted_by(self.client.find_similar, {
            "exclude_source_domain": exclude_source_domain,
            **filters,
        }))

        similar_response = [
            {"href": result.url, "body": result.text or ""} for result in results.results
        ]
        return similar_response

    def get_contents(self, ids, **options):
        """
        Retrieves the contents of the specified IDs using the Exa API.
        Args:
            ids: The IDs of the documents to retrieve.
            **options: Additional options for content retrieval.
        Returns:
            A list of document contents.
        """
        results = self.client.get_contents(ids, **options)

        contents_response = [
            {"id": result.id, "content": result.text} for result in results.results
        ]
        return contents_response
