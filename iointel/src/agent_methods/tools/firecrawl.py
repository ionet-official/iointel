import asyncio
import os
from typing import Optional

from firecrawl import FirecrawlApp
from iointel.src.utilities.decorators import register_tool
from pydantic import BaseModel

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")


class FirecrawlResponse(BaseModel):
    markdown: str
    metadata: dict


class Crawler(BaseModel):
    """
    A wrapper class for the FirecrawlApp that provides methods for scraping,
    crawling, mapping, extracting, and watching crawl jobs.
    """

    api_key: str
    timeout: int
    _app: FirecrawlApp | None = None

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60) -> None:
        """
        Initialize the Firecrawl app.
        Args:
            api_key (str): The API key for Firecrawl.
            timeout (int): How many seconds to wait while scraping.
        """
        if not api_key:
            api_key = FIRECRAWL_API_KEY
        if not FIRECRAWL_API_KEY:
            raise RuntimeError("Firecrawl API key is not set")
        super().__init__(api_key=api_key, timeout=timeout)
        self._app = FirecrawlApp(api_key=api_key)

    @register_tool
    def scrape_url(self, url: str, timeout: int | None = None) -> FirecrawlResponse:
        """
        Scrape a single URL.
        Args:
            url (str): The URL to scrape
            timeout (int): How many seconds to wait while scraping.
        Returns:
            Dict[str, Any]: The scraping result.
        """
        # firecrawl uses ms for timeout units
        response = self._app.scrape_url(url, timeout=(timeout or self.timeout) * 1000)
        return FirecrawlResponse(markdown=response.markdown, metadata=response.metadata)

    @register_tool
    async def async_scrape_url(
        self, url: str, timeout: int | None = None
    ) -> FirecrawlResponse:
        """
        Scrape a single URL.
        Args:
            url (str): The URL to scrape.
            timeout (int): How many seconds to wait while scraping
        Returns:
            Dict[str, Any]: The scraping result.
        """
        return await asyncio.to_thread(self.scrape_url, url, timeout)
