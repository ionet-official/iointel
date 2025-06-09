from typing import Any, Optional
from agno.tools.baidusearch import BaiduSearchTools as AgnoBaiduSearchTools
from pydantic import Field

from .common import make_base, wrap_tool


class BaiduSearch(make_base(AgnoBaiduSearchTools)):
    fixed_max_results: Optional[int] = Field(default=None, frozen=True)
    fixed_language: Optional[str] = Field(default=None, frozen=True)
    headers: Optional[Any] = Field(default=None, frozen=True)
    proxy: Optional[str] = Field(default=None, frozen=True)
    timeout: Optional[int] = Field(default=10, frozen=True)
    debug: Optional[bool] = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            fixed_max_results=self.fixed_max_results,
            fixed_language=self.fixed_language,
            headers=self.headers,
            proxy=self.proxy,
            timeout=self.timeout,
            debug=self.debug,
        )

    @wrap_tool("agno__baidu__make_request", AgnoBaiduSearchTools.baidu_search)
    def baidu_search(
        self, query: str, max_results: int = 5, language: str = "zh"
    ) -> str:
        return self._tool.baidu_search(query, max_results, language)
