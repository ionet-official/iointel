from os import getenv
from typing import Optional
from agno.tools.jina import JinaReaderTools as AgnoJinaReaderTools
from .common import make_base, wrap_tool
from pydantic import Field


class JinaReader(make_base(AgnoJinaReaderTools)):
    api_key: Optional[str] = Field(default=getenv("JINA_API_KEY"), frozen=True)
    base_url: str = Field(default="https://r.jina.ai/", frozen=True)
    search_url: str = Field(default="https://s.jina.ai/", frozen=True)
    max_content_length: int = Field(default=10000, frozen=True)
    timeout: Optional[int] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            base_url=self.base_url_,
            search_url=self.search_url_,
            max_content_length=self.max_content_length_,
            timeout=self.timeout_,
            read_url=True,
            search_query=True,
        )

    @wrap_tool("agno__jinareader__read_url", AgnoJinaReaderTools.read_url)
    def read_url(self, url: str) -> str:
        return self._tool.read_url(self, url)

    @wrap_tool("agno__jinareader__search_query", AgnoJinaReaderTools.search_query)
    def search_query(self, query: str) -> str:
        return self._tool.search_query(self, query)
