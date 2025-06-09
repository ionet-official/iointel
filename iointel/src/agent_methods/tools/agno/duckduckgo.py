from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools

from .common import make_base, wrap_tool


class DuckDuckGo(make_base(AgnoDuckDuckGoTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("agno_ddg_duckduckgo_search", AgnoDuckDuckGoTools.duckduckgo_search)
    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.duckduckgo_search(query, max_results)

    @wrap_tool("agno_ddg_duckduckgo_news", AgnoDuckDuckGoTools.duckduckgo_news)
    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        return self._tool.duckduckgo_news(query, max_results)
