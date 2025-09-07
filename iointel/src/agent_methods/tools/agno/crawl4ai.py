from typing import Optional
from agno.tools.crawl4ai import Crawl4aiTools as AgnoCrawl4aiTools
from pydantic import Field
from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool


class Crawl4ai(make_base(AgnoCrawl4aiTools)):
    max_length: Optional[int] = Field(default=1000, frozen=True)

    def _get_tool(self):
        return self.Inner(max_length=self.max_length)

    @wrap_tool("agno__crawl4ai__web_crawler", AgnoCrawl4aiTools.web_crawler)
    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:
        return self._tool.web_crawler(url, max_length)
