from pathlib import Path
from typing import Optional
from agno.tools.arxiv import ArxivTools as AgnoArxivTools

from .common import make_base, wrap_tool


class Arxiv(make_base(AgnoArxivTools)):
    download_dir: Path | None = None

    def _get_tool(self):
        # TODO: define a better default option for downloading pdfs, agno arxiv has a weird choice
        return self.Inner(download_dir=self.download_dir)

    @wrap_tool("arxiv_search", AgnoArxivTools.search_arxiv_and_return_articles)
    def search_arxiv_and_return_articles(
        self, query: str, num_articles: int = 10
    ) -> str:
        return self._tool.search_arxiv_and_return_articles(query, num_articles)

    @wrap_tool("arxiv_read_papers", AgnoArxivTools.read_arxiv_papers)
    def read_arxiv_papers(
        self, id_list: list[str], pages_to_read: Optional[int] = None
    ) -> str:
        return self._tool.read_arxiv_papers(id_list, pages_to_read)
