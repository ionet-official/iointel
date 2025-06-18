from typing import Any, Optional, Dict
from agno.tools.newspaper4k import Newspaper4kTools as AgnoNewspaper4kTools
from .common import make_base, wrap_tool
from pydantic import Field


class Newspaper4k(make_base(AgnoNewspaper4kTools)):
    # read_article: bool = Field(default=True, frozen=True)
    include_summary: bool = Field(default=False, frozen=True)
    article_length: Optional[int] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            read_article=True,
            include_summary=self.include_summary,
            article_length=self.article_length,
        )

    @wrap_tool(
        "agno__newspaper4k__get_article_data", AgnoNewspaper4kTools.get_article_data
    )
    def get_article_data(self, url: str) -> Optional[Dict[str, Any]]:
        return self._tool.get_article_data(url)

    @wrap_tool("agno__newspaper4k__read_article", AgnoNewspaper4kTools.read_article)
    def read_article(self, url: str) -> str:
        return self._tool.read_article(url)
