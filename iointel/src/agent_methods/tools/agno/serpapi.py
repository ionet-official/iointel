from agno.tools.serpapi import SerpApiTools as AgnoSerpApiTools
from .common import make_base, wrap_tool


class SerpApi(make_base(AgnoSerpApiTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            search_youtube=self.search_youtube,
        )

    @wrap_tool("agno__serpapi__search_google", AgnoSerpApiTools.search_google)
    def search_google(self, query: str, num_results: int = 10) -> str:
        return self.search_google(self, query, num_results)

    @wrap_tool("agno__serpapi__search_youtube", AgnoSerpApiTools.search_youtube)
    def search_youtube(self, query: str) -> str:
        return self.search_youtube(self, query)
