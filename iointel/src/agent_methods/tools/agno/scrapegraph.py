from typing import Optional
from agno.tools.scrapegraph import ScrapeGraphTools as AgnoScrapeGraphTools
from .common import make_base, wrap_tool
from pydantic import Field


class ScrapeGraph(make_base(AgnoScrapeGraphTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            smartscraper=True,
            markdownify=True,
        )

    @wrap_tool("agno__scrapegraph__smartscraper", AgnoScrapeGraphTools.smartscraper)
    def smartscraper(self, url: str, prompt: str) -> str:
        return self._tool.smartscraper(self, url, prompt)

    @wrap_tool("agno__scrapegraph__markdownify", AgnoScrapeGraphTools.markdownify)
    def markdownify(self, url: str) -> str:
        return self._tool.markdownify(self, url)
