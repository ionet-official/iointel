import asyncio
import os
from typing import Optional

from firecrawl import FirecrawlApp
from iointel.src.utilities.decorators import register_tool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv("creds.env")

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
    operational_fields: list[str] = ["statusCode", "contentType", "proxyUsed", "creditsUsed"]
    _app: FirecrawlApp | None = None

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60) -> None:
        """
        Initialize the Firecrawl app.
        Args:
            api_key (str): The API key for Firecrawl.
            timeout (int): How many seconds to wait while scraping.
        """
        if not api_key:
            if not FIRECRAWL_API_KEY:
                raise RuntimeError("Firecrawl API key is not set")
            api_key = FIRECRAWL_API_KEY
        
        super().__init__(api_key=api_key, timeout=timeout)
        self._app = FirecrawlApp(api_key=api_key)

    def _result_to_llm_input(self, result: FirecrawlResponse, with_operational: bool=False) -> dict:
        """
        Convert a FirecrawlResponse to an LLM input.
        """
        operational_info = {
            k: result.metadata[k]
            for k in self.operational_fields    
        }
        pruned_metadata = {
            k: result.metadata[k]
            for k in result.metadata.keys() if k not in self.operational_fields
        }
        if with_operational:
            llm_input = {
                "markdown": result.markdown,
                "metadata": pruned_metadata,
                "operational_info": operational_info,
            }
        else:
            llm_input = {
                "markdown": result.markdown,
                "metadata": pruned_metadata,
            }
        return llm_input

    @register_tool
    def scrape_url(self, url: str, timeout: int | None = None, with_operational: bool=False) -> FirecrawlResponse:
        f"""
        Scrape a single URL.
        Args:
            url (str): The URL to scrape
            timeout (int): How many seconds to wait while scraping. Default is 10 seconds.
            with_operational (bool): Whether to include operational information in the result (e.g. {self.operational_fields})
        Returns:
            Dict[str, Any]: The scraping result.
        """
        # firecrawl uses ms for timeout units
        response = self._app.scrape_url(url, timeout=(timeout or self.timeout) * 1000)
        result =  FirecrawlResponse(markdown=response.markdown, metadata=response.metadata)
        return self._result_to_llm_input(result, with_operational)

    @register_tool
    async def async_scrape_url(
        self, url: str, timeout: int | None = None, with_operational: bool=False
    ) -> FirecrawlResponse:
        """
        Scrape a single URL.
        Args:
            url (str): The URL to scrape.
            timeout (int): How many seconds to wait while scraping
            with_operational (bool): Whether to include operational information in the result (e.g. status code, credits used, etc.)
        Returns:
            Dict[str, Any]: The scraping result.
        """
        return await asyncio.to_thread(self.scrape_url, url, timeout, with_operational)


    @register_tool
    def crawl_url(self, url: str, max_depth: int | None = None, limit: int | None = None, with_operational: bool=True) -> FirecrawlResponse:
        """
        Crawl a single URL.
        Args:
            url (str): The URL to crawl.
            max_depth (int): Maximum depth to crawl (default: None for unlimited).
            limit (int): Maximum number of pages to crawl (default: None for unlimited).
            with_operational (bool): Whether to include operational information in the result (e.g. status code, credits used, etc.)
        Returns:
            Dict[str, Any]: The crawling result.
        """
        # Pass timeout through scrape_options instead of as direct parameter
        from firecrawl.firecrawl import ScrapeOptions
        scrape_options = ScrapeOptions(timeout=self.timeout * 1000)  # Convert to ms
        
        response = self._app.crawl_url(
            url, 
            max_depth=max_depth, 
            limit=limit,
            scrape_options=scrape_options
        )
        result = FirecrawlResponse(markdown=response.markdown, metadata=response.metadata)
        return self._result_to_llm_input(result, with_operational)