from typing import List, Optional
from agno.tools.searxng import Searxng as AgnoSearxng
from .common import make_base, wrap_tool
from pydantic import Field


class Searxng(make_base(AgnoSearxng)):
    host: str = Field(frozen=True)
    engines: List[str] = Field(default=[], frozen=True)
    fixed_max_results: Optional[int] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            host=self.host,
            engines=self.engines,
            fixed_max_results=self.fixed_max_results,
            images=True,
            it=True,
            map=True,
            music=True,
            news=True,
            science=True,
            videos=True,
        )

    @wrap_tool("agno__searxng__search", AgnoSearxng.search)
    def search(self, query: str, max_results: int = 5) -> str:
        return self._tool.search(query, max_results)

    @wrap_tool("agno__searxng__image_search", AgnoSearxng.image_search)
    def image_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.image_search(query, max_results)

    @wrap_tool("agno__searxng__it_search", AgnoSearxng.it_search)
    def it_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.it_search(query, max_results)

    @wrap_tool("agno__searxng__map_search", AgnoSearxng.map_search)
    def map_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.map_search(query, max_results)

    @wrap_tool("agno__searxng__music_search", AgnoSearxng.music_search)
    def music_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.music_search(query, max_results)

    @wrap_tool("agno__searxng__news_search", AgnoSearxng.news_search)
    def news_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.news_search(query, max_results)

    @wrap_tool("agno__searxng__science_search", AgnoSearxng.science_search)
    def science_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.science_search(query, max_results)

    @wrap_tool("agno__searxng__video_search", AgnoSearxng.video_search)
    def video_search(self, query: str, max_results: int = 5) -> str:
        return self._tool.video_search(query, max_results)
