from agno.tools.tavily import TavilyTools as AgnoTavilyTools
from .common import make_base, wrap_tool


class Tavily(make_base(AgnoTavilyTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            search=self.search_,
            max_tokens=self.max_tokens_,
            include_answer=self.include_answer_,
            search_depth=self.search_depth_,
            format=self.format_,
            use_search_context=self.use_search_context_,
        )

    @wrap_tool(
        "agno__tavily__web_search_using_tavily", AgnoTavilyTools.web_search_using_tavily
    )
    def web_search_using_tavily(self, query: str, max_results: int = 5) -> str:
        return self.web_search_using_tavily(self, query, max_results)

    @wrap_tool(
        "agno__tavily__web_search_with_tavily", AgnoTavilyTools.web_search_with_tavily
    )
    def web_search_with_tavily(self, query: str) -> str:
        return self.web_search_with_tavily(self, query)
