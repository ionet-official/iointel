from typing import Any, Optional, Dict
from agno.tools.newspaper4k import Newspaper4kTools as AgnoNewspaper4kTools
from .common import make_base, wrap_tool


class Newspaper4k(make_base(AgnoNewspaper4kTools)):
    def _get_tool(self):
        return self.Inner(
            read_article=self.read_article,
            include_summary=self.include_summary_,
            article_length=self.article_length_,
        )

    @wrap_tool(
        "agno__newspaper4k__get_article_data", AgnoNewspaper4kTools.get_article_data
    )
    def get_article_data(self, url: str) -> Optional[Dict[str, Any]]:
        return self.get_article_data(self, url)

    @wrap_tool("agno__newspaper4k__read_article", AgnoNewspaper4kTools.read_article)
    def read_article(self, url: str) -> str:
        return self.read_article(self, url)
