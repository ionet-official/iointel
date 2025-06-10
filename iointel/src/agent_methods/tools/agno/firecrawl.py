from typing import Any, Dict, List, Optional
from agno.tools.firecrawl import FirecrawlTools as AgnoFirecrawTools
from pydantic import Field
from .common import make_base, wrap_tool


class Firecrawl(make_base(AgnoFirecrawTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    formats: Optional[List[str]] = Field(default=None, frozen=True)
    limit: int = Field(default=10, frozen=True)
    poll_interval: int = Field(default=30, frozen=True)
    scrape: bool = Field(default=True, frozen=True)
    crawl: bool = Field(default=False, frozen=True)
    mapping: bool = Field(default=False, frozen=True)
    search_: bool = Field(default=False, frozen=True)
    search_params: Optional[Dict[str, Any]] = Field(default=None, frozen=True)
    api_url: Optional[str] = Field(default="https://api.firecrawl.dev", frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            formats=self.formats,
            limit=self.limit,
            poll_interval=self.poll_interval,
            scrape=self.scrape,
            crawl=self.crawl,
            mapping=self.mapping,
            search=self.search_,
            search_params=self.search_params,
            api_url=self.api_url,
        )

    @wrap_tool("agno__firecrawl__scrape_website", AgnoFirecrawTools.scrape_website)
    def scrape_website(self, url: str) -> str:
        return self._tool.scrape_website(url)

    @wrap_tool("agno__firecrawl__crawl_website", AgnoFirecrawTools.crawl_website)
    def crawl_website(self, url: str, limit: Optional[int] = None) -> str:
        return self._tool.crawl_website(url, limit)

    @wrap_tool("agno__firecrawl__map_website", AgnoFirecrawTools.map_website)
    def map_website(self, url: str) -> str:
        return self._tool.map_website(url)

    @wrap_tool("agno__firecrawl__search", AgnoFirecrawTools.search)
    def search(self, query: str, limit: Optional[int] = None):
        return self._tool.search(query, limit)
