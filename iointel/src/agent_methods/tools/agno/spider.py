from typing import Optional
from agno.tools.spider import SpiderTools as AgnoSpiderTools
from .common import make_base, wrap_tool


class Spider(make_base(AgnoSpiderTools)):
    def _get_tool(self):
        return self.Inner(
            max_results=self.max_results_,
            url=self.url_,
            optional_params=self.optional_params_,
        )

    @wrap_tool("agno__spider__search", AgnoSpiderTools.search)
    def search(self, query: str, max_results: int = 5) -> str:
        return self.search(self, query, max_results)

    @wrap_tool("agno__spider__scrape", AgnoSpiderTools.scrape)
    def scrape(self, url: str) -> str:
        return self.scrape(self, url)

    @wrap_tool("agno__spider__crawl", AgnoSpiderTools.crawl)
    def crawl(self, url: str, limit: Optional[int] = None) -> str:
        return self.crawl(self, url, limit)

    @wrap_tool("agno__spider___search", AgnoSpiderTools._search)
    def _search(self, query: str, max_results: int = 1) -> str:
        return self._search(self, query, max_results)

    @wrap_tool("agno__spider___scrape", AgnoSpiderTools._scrape)
    def _scrape(self, url: str) -> str:
        return self._scrape(self, url)

    @wrap_tool("agno__spider___crawl", AgnoSpiderTools._crawl)
    def _crawl(self, url: str, limit: Optional[int] = None) -> str:
        return self._crawl(self, url, limit)
