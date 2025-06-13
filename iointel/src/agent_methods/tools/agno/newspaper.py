from agno.tools.newspaper import NewspaperTools as AgnoNewspaperTools
from .common import make_base, wrap_tool


class Newspaper(make_base(AgnoNewspaperTools)):
    def _get_tool(self):
        return self.Inner(
            get_article_text=self.get_article_text,
        )

    @wrap_tool("agno__newspaper__get_article_text", AgnoNewspaperTools.get_article_text)
    def get_article_text(self, url: str) -> str:
        return self.get_article_text(self, url)
