import os
from typing import Optional, List, Dict, Any
import httpx
from iointel.src.utilities.decorators import register_tool
from pydantic import BaseModel


class SearchResult(BaseModel):
    url: str
    title: str
    content: str


class InfoboxUrl(BaseModel):
    title: str
    url: str


class Infobox(BaseModel):
    infobox: str
    id: str
    content: str
    urls: List[InfoboxUrl]


class SearchResponse(BaseModel):
    query: str
    number_of_results: int
    results: List[SearchResult]
    infoboxes: List[Infobox]


# ---------------------------
# SearxngClient Class with Internal Pagination
# ---------------------------

# Here's how to run Searxng locally
# export PORT=8080
# docker pull searxng/searxng
# docker run --rm -d \
#    -p ${PORT}:8080 \
#    -v "${PWD}/searxng:/etc/searxng" \
#    -e "BASE_URL=http://localhost:$PORT/" \
#    -e "INSTANCE_NAME=my-instance" \
#    searxng/searxng


class SearxngClient(BaseModel):
    """
    A client for interacting with the SearxNG search API.

    This client provides both asynchronous and synchronous search methods.
    Pagination is handled internally if the caller specifies more than one page
    via the 'pages' parameter.
    """

    base_url: str
    timeout: int
    _client: httpx.Client

    def __init__(self, base_url: Optional[str] = None, timeout: int = 10) -> None:
        """
        Initialize the SearxngClient.

        Args:
            base_url (Optional[str]): The base URL of the SearxNG instance.
                Defaults to the environment variable 'SEARXNG_URL' or "http://localhost:8081".
            timeout (int): Timeout for HTTP requests in seconds.
        """
        base_url = base_url or os.getenv("SEARXNG_URL")
        if not base_url:
            raise RuntimeError(
                "Searxng base url is not set in SEARXNG_URL env variable"
            )
        super().__init__(base_url=base_url, timeout=timeout)
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    @register_tool("searxng_search")
    def search(self, query: str, pages: int = 1) -> SearchResponse:
        """
        Asynchronously perform a search query using the SearxNG API.
        If 'pages' is greater than 1, the method iterates over pages 1..pages
        and combines the results.

        Args:
            query (str): The search query.
            pages (int): The number of pages to retrieve (default is 1).

        Returns:
            SearchResponse: A combined search response containing results and infoboxes from all pages.

        Raises:
            httpx.HTTPError: If any HTTP request fails.
        """
        combined_results: List[SearchResult] = []
        combined_infoboxes: List[Infobox] = []
        total_results = 0

        for pageno in range(1, pages + 1):
            params: Dict[str, Any] = {
                "q": query,
                "format": "json",
                "pageno": str(pageno),
            }
            response = self._client.get("/search", params=params)
            response.raise_for_status()
            page_data = SearchResponse.model_validate_json(response.text)
            combined_results.extend(page_data.results)
            combined_infoboxes.extend(page_data.infoboxes)
            total_results += page_data.number_of_results

        return SearchResponse(
            query=query,
            number_of_results=total_results,
            results=combined_results,
            infoboxes=combined_infoboxes,
        )

    @register_tool("searxng_get_urls")
    def get_urls(self, query: str, pages: int = 1) -> List[str]:
        """
        Synchronously perform a search query using the SearxNG API.
        If 'pages' is greater than 1, the method iterates over pages 1..pages
        and combines the results.

        Args:
            query (str): The search query.
            pages (int): The number of pages to retrieve (default is 1).

        Returns:
            List[str]: A list of URLs from the combined search results.

        Raises:
            httpx.HTTPError: If any HTTP request fails.
        """
        return [result.url for result in (self.search(query, pages)).results]

    def close(self) -> None:
        """
        Close the underlying HTTP client.
        """
        if hasattr(self, '_client') and self._client:
            self._client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass
