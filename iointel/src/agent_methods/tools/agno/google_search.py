from typing import Any, Optional
from agno.tools.googlesearch import GoogleSearchTools as AgnoGoogleSearchTools
from .common import make_base, wrap_tool
from pydantic import Field


class GoogleSearch(make_base(AgnoGoogleSearchTools)):
    fixed_max_results: Optional[int] = Field(default=None, frozen=True)
    fixed_language: Optional[str] = Field(default=None, frozen=True)
    headers: Optional[Any] = Field(default=None, frozen=True)
    proxy: Optional[str] = Field(default=None, frozen=True)
    timeout: Optional[int] = Field(default=10, frozen=True)

    def _get_tool(self):
        return self._tool.Inner(
            fixed_max_results=self.fixed_max_results,
            fixed_language=self.fixed_language,
            headers=self.headers,
            proxy=self.proxy,
            timeout=self.timeout,
        )

    @wrap_tool(
        "agno__google_search__google_search", AgnoGoogleSearchTools.google_search
    )
    def google_search(
        self, query: str, max_results: int = 5, language: str = "en"
    ) -> str:
        return self._tool.google_search(query, max_results, language)
