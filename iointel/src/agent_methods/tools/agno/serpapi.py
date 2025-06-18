from typing import Optional
from agno.tools.serpapi import SerpApiTools as AgnoSerpApiTools
from .common import make_base, wrap_tool
from pydantic import Field


class SerpApi(make_base(AgnoSerpApiTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            search_youtube=True,
        )

    @wrap_tool("agno__serpapi__search_google", AgnoSerpApiTools.search_google)
    def search_google(self, query: str, num_results: int = 10) -> str:
        return self.search_google(query, num_results)

    @wrap_tool("agno__serpapi__search_youtube", AgnoSerpApiTools.search_youtube)
    def search_youtube(self, query: str) -> str:
        return self.search_youtube(query)
