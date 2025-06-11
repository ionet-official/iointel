from typing import Optional
from agno.tools.searxng import Searxng as AgnoSearxng
from .common import make_base, wrap_tool


class Se(make_base(AgnoSearxng)):
    def _get_tool(self):
        return self.Inner(
            host=self.host_,
            engines=self.engines_,
            fixed_max_results=self.fixed_max_results_,
            images=self.images_,
            it=self.it_,
            map=self.map_,
            music=self.music_,
            news=self.news_,
            science=self.science_,
            videos=self.videos_,
        )

    @wrap_tool("agno__se__search", AgnoSearxng.search)
    def search(self, query: str, max_results: int = 5) -> str:
        return self.search(self, query, max_results)

    @wrap_tool("agno__se__image_search", AgnoSearxng.image_search)
    def image_search(self, query: str, max_results: int = 5) -> str:
        return self.image_search(self, query, max_results)

    @wrap_tool("agno__se__it_search", AgnoSearxng.it_search)
    def it_search(self, query: str, max_results: int = 5) -> str:
        return self.it_search(self, query, max_results)

    @wrap_tool("agno__se__map_search", AgnoSearxng.map_search)
    def map_search(self, query: str, max_results: int = 5) -> str:
        return self.map_search(self, query, max_results)

    @wrap_tool("agno__se__music_search", AgnoSearxng.music_search)
    def music_search(self, query: str, max_results: int = 5) -> str:
        return self.music_search(self, query, max_results)

    @wrap_tool("agno__se__news_search", AgnoSearxng.news_search)
    def news_search(self, query: str, max_results: int = 5) -> str:
        return self.news_search(self, query, max_results)

    @wrap_tool("agno__se__science_search", AgnoSearxng.science_search)
    def science_search(self, query: str, max_results: int = 5) -> str:
        return self.science_search(self, query, max_results)

    @wrap_tool("agno__se__video_search", AgnoSearxng.video_search)
    def video_search(self, query: str, max_results: int = 5) -> str:
        return self.video_search(self, query, max_results)

    @wrap_tool("agno__se___search", AgnoSearxng._search)
    def _search(
        self, query: str, category: Optional[str] = None, max_results: int = 5
    ) -> str:
        return self._search(self, query, category, max_results)
