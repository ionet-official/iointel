from pathlib import Path
from typing import Optional
from functools import wraps
from agno.tools.arxiv import ArxivTools as AgnoArxivTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class Arxiv(DisableAgnoRegistryMixin, AgnoArxivTools):
    def __init__(self, download_dir: Optional[Path] = None):
        # TODO: define a better default option for downloading pdfs, agno arxiv has a weird choice
        super().__init__(download_dir=download_dir)

    @register_tool(name="arxiv_search")
    @wraps(AgnoArxivTools.search_arxiv_and_return_articles)
    def search_arxiv_and_return_articles(
        self, query: str, num_articles: int = 10
    ) -> str:
        return super().search_arxiv_and_return_articles(query, num_articles)

    @register_tool(name="arxiv_read_papers")
    @wraps(AgnoArxivTools.read_arxiv_papers)
    def read_arxiv_papers(
        self, id_list: list[str], pages_to_read: Optional[int] = None
    ) -> str:
        return super().read_arxiv_papers(id_list, pages_to_read)
