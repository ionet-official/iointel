from typing import Optional
from agno.tools.crawl4ai import Crawl4aiTools as AgnoCrawl4aiTools

from .common import make_base, wrap_tool


class Crawl4ai(make_base(AgnoCrawl4aiTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("agno__crawl4ai__web_crawler", AgnoCrawl4aiTools.web_crawler)
    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:
        return self._tool.web_crawler(url, max_length)
