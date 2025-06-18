from agno.tools.hackernews import HackerNewsTools as AgnoHackerNewsTools
from .common import make_base, wrap_tool


class HackerNews(make_base(AgnoHackerNewsTools)):
    def _get_tool(self):
        return self.Inner(
            get_top_stories=True,
            get_user_details=True,
        )

    @wrap_tool(
        "agno__hackernews__get_top_hackernews_stories",
        AgnoHackerNewsTools.get_top_hackernews_stories,
    )
    def get_top_hackernews_stories(self, num_stories: int = 10) -> str:
        return self._tool.get_top_hackernews_stories(num_stories)

    @wrap_tool(
        "agno__hackernews__get_user_details", AgnoHackerNewsTools.get_user_details
    )
    def get_user_details(self, username: str) -> str:
        return self._tool.get_user_details(self, username)
