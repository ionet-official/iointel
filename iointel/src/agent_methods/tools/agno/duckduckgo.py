from typing import Any, Optional
from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools
from pydantic import Field

from .common import make_base, wrap_tool


class DuckDuckGo(make_base(AgnoDuckDuckGoTools)):
    modifier: Optional[str] = Field(default=None, frozen=True)
    fixed_max_results: Optional[int] = Field(default=None, frozen=True)
    headers: Optional[Any] = Field(default=None, frozen=True)
    proxy: Optional[str] = Field(default=None, frozen=True)
    proxies: Optional[Any] = Field(default=None, frozen=True)
    timeout: Optional[int] = Field(default=10, frozen=True)
    verify_ssl: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            search=True,
            news=True,
            modifier=self.modifier,
            fixed_max_results=self.fixed_max_results,
            headers=self.headers,
            proxy=self.proxy,
            proxies=self.proxies,
            timeout=self.timeout,
            verify_ssl=self.verify_ssl,
        )

    @wrap_tool("agno_ddg_duckduckgo_search", AgnoDuckDuckGoTools.duckduckgo_search)
    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.duckduckgo_search(query, max_results)

    @wrap_tool("agno_ddg_duckduckgo_news", AgnoDuckDuckGoTools.duckduckgo_news)
    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        return self._tool.duckduckgo_news(query, max_results)
