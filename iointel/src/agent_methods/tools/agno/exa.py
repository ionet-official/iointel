# from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools
from typing import Optional
from agno.tools.exa import ExaTools as AgnoExaTools
from .common import make_base, wrap_tool


class Exa(make_base(AgnoExaTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("agno_exa_search_exa", AgnoExaTools.search_exa)
    def search_exa(
        self, query: str, num_results: int = 5, category: Optional[str] = None
    ) -> str:
        return self._tool.search_exa(query, num_results, category)

    @wrap_tool("agno_exa_get_contents", AgnoExaTools.get_contents)
    def get_contents(self, urls: list[str]) -> str:
        return self._tool.get_contents(urls)

    @wrap_tool("agno_exa_find_similar", AgnoExaTools.find_similar)
    def find_similar(self, url: str, num_results: int = 5) -> str:
        return self._tool.find_similar(url, num_results)

    @wrap_tool("agno_exa_exa_answer", AgnoExaTools.exa_answer)
    def exa_answer(self, query: str, text: bool = False) -> str:
        return self._tool.exa_answer(query, text)
