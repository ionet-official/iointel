from agno.tools.scrapegraph import ScrapeGraphTools as AgnoScrapeGraphTools
from .common import make_base, wrap_tool


class ScrapeGraph(make_base(AgnoScrapeGraphTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            smartscraper=self.smartscraper_,
            markdownify=self.markdownify,
        )

    @wrap_tool("agno__scrapegraph__smartscraper", AgnoScrapeGraphTools.smartscraper)
    def smartscraper(self, url: str, prompt: str) -> str:
        return self.smartscraper(self, url, prompt)

    @wrap_tool("agno__scrapegraph__markdownify", AgnoScrapeGraphTools.markdownify)
    def markdownify(self, url: str) -> str:
        return self.markdownify(self, url)
