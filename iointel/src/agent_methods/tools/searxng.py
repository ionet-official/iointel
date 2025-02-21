import os
from typing import Optional, List, Dict, Any
import httpx  # For synchronous requests
from httpx import AsyncClient  # For asynchronous requests
from pydantic import BaseModel


# ---------------------------
# Pydantic Models Definitions
# ---------------------------


class SearchResult(BaseModel):
    url: str
    title: str
    content: str


class URL(BaseModel):
    url: str


class URLs(BaseModel):
    urls: List[URL]


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


class SearxngClient:
    """
    A client for interacting with the SearxNG search API.

    This client provides both asynchronous and synchronous search methods.
    Pagination is handled internally if the caller specifies more than one page
    via the 'pages' parameter.

    Example usage (asynchronous):
        import asyncio

        async def main():
            client = SearxngClient()
            # Single-page search (pages=1 by default)
            response = await client.search("python programming")
            print("Single Page Response:", response)
            # Paginated search: pages 1 to 3
            paginated_response = await client.search("python programming", pages=3)
            print("Paginated Response:", paginated_response)
            await client.close()

        asyncio.run(main())

    Example usage (synchronous):
        client = SearxngClient()
        # Single-page search
        response = client.search_sync("python programming")
        print("Single Page Response:", response)
        # Paginated search: pages 1 to 3
        paginated_response = client.search_sync("python programming", pages=3)
        print("Paginated Response:", paginated_response)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 10) -> None:
        """
        Initialize the SearxngClient.

        Args:
            base_url (Optional[str]): The base URL of the SearxNG instance.
                Defaults to the environment variable 'SEARXNG_URL' or "http://localhost:8081".
            timeout (int): Timeout for HTTP requests in seconds.
        """
        self.base_url = base_url or os.getenv("SEARXNG_URL", "http://localhost:8081")
        self.timeout = timeout
        self.async_client = AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def search(self, query: str, pages: int = 1) -> SearchResponse:
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
            response = await self.async_client.get("/search", params=params)
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

    def search_sync(self, query: str, pages: int = 1) -> SearchResponse:
        """
        Synchronously perform a search query using the SearxNG API.
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

        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            for pageno in range(1, pages + 1):
                params: Dict[str, Any] = {
                    "q": query,
                    "format": "json",
                    "pageno": str(pageno),
                }
                response = client.get("/search", params=params)
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
        response = self.search_sync(query, pages)
        urls = URLs(urls=[URL(url=result.url) for result in response.results])
        return [result.url for result in urls.urls]

    async def get_urls_async(self, query: str, pages: int = 1) -> List[str]:
        """
        Asynchronously perform a search query using the SearxNG API.
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
        response = await self.search(query, pages)
        urls = URLs(urls=[URL(url=result.url) for result in response.results])
        return [result.url for result in urls.urls]

    async def close(self) -> None:
        """
        Close the underlying asynchronous HTTP client.
        """
        await self.async_client.aclose()
