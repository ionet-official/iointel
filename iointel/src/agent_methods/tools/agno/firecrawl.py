from typing import Optional
from agno.tools.firecrawl import FirecrawlTools as AgnoFirecrawTools

from .common import make_base, wrap_tool


class Firecrawl(make_base(AgnoFirecrawTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

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
