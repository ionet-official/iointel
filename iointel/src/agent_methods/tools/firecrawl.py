import asyncio
import os
from typing import Optional
from typing import TypeVar

from firecrawl import FirecrawlApp
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")


class FirecrawlResponse(BaseModel):
    markdown: str
    metadata: dict


class Crawler:
    """
    A wrapper class for the FirecrawlApp that provides methods for scraping,
    crawling, mapping, extracting, and watching crawl jobs.
    """

    def __init__(self, api_key: str = None, version: Optional[str] = None) -> None:
        """
        Initialize the Firecrawl app.
        Args:
            api_key (str): The API key for Firecrawl.
            version (Optional[str]): Optional API version.
        """
        if not api_key:
            api_key = FIRECRAWL_API_KEY
        if not FIRECRAWL_API_KEY:
            raise RuntimeError("Firecrawl API key is not set")
        if version:
            self.app: FirecrawlApp = FirecrawlApp(api_key=api_key, version=version)
        else:
            self.app: FirecrawlApp = FirecrawlApp(api_key=api_key)

    def scrape_url(self, url: str) -> FirecrawlResponse:
        """
        Scrape a single URL.
        Args:
            url (str): The URL to scrape
        Returns:
            Dict[str, Any]: The scraping result.
        """
        response = self.app.scrape_url(url)
        return FirecrawlResponse(markdown=response.markdown, metadata=response.metadata)

    async def async_scrape_url(self, url: str) -> FirecrawlResponse:
        """
        Scrape a single URL.
        Args:
            url (str): The URL to scrape.
        Returns:
            Dict[str, Any]: The scraping result.
        """
        return await asyncio.to_thread(self.scrape_url, url)
