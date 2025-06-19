from typing import Optional, TYPE_CHECKING
from agno.tools.reddit import RedditTools as AgnoRedditTools
from .common import make_base, wrap_tool

if TYPE_CHECKING:
    import praw
from pydantic import Field


class Reddit(make_base(AgnoRedditTools)):
    reddit_instance: Optional[praw.Reddit] = Field(default=None, frozen=True)
    client_id: Optional[str] = Field(default=None, frozen=True)
    client_secret: Optional[str] = Field(default=None, frozen=True)
    user_agent: Optional[str] = Field(default=None, frozen=True)
    username: Optional[str] = Field(default=None, frozen=True)
    password: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            reddit_instance=self.reddit_instance,
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            username=self.username,
            password=self.password,
            get_user_info=True,
            get_top_posts=True,
            get_subreddit_info=True,
            get_trending_subreddits=True,
            get_subreddit_stats=True,
            create_post=True,
            reply_to_post=True,
            reply_to_comment=True,
        )

    @wrap_tool("agno__reddit__get_user_info", AgnoRedditTools.get_user_info)
    def get_user_info(self, username: str) -> str:
        return self._tool.get_user_info(self, username)

    @wrap_tool("agno__reddit__get_top_posts", AgnoRedditTools.get_top_posts)
    def get_top_posts(
        self, subreddit: str, time_filter: str = "week", limit: int = 10
    ) -> str:
        return self._tool.get_top_posts(self, subreddit, time_filter, limit)

    @wrap_tool("agno__reddit__get_subreddit_info", AgnoRedditTools.get_subreddit_info)
    def get_subreddit_info(self, subreddit_name: str) -> str:
        return self._tool.get_subreddit_info(self, subreddit_name)

    @wrap_tool(
        "agno__reddit__get_trending_subreddits", AgnoRedditTools.get_trending_subreddits
    )
    def get_trending_subreddits(self) -> str:
        return self._tool.get_trending_subreddits(self)

    @wrap_tool("agno__reddit__get_subreddit_stats", AgnoRedditTools.get_subreddit_stats)
    def get_subreddit_stats(self, subreddit: str) -> str:
        return self._tool.get_subreddit_stats(self, subreddit)

    @wrap_tool("agno__reddit__create_post", AgnoRedditTools.create_post)
    def create_post(
        self,
        subreddit: str,
        title: str,
        content: str,
        flair: Optional[str] = None,
        is_self: bool = True,
    ) -> str:
        return self._tool.create_post(self, subreddit, title, content, flair, is_self)

    @wrap_tool("agno__reddit__reply_to_post", AgnoRedditTools.reply_to_post)
    def reply_to_post(
        self, post_id: str, content: str, subreddit: Optional[str] = None
    ) -> str:
        return self._tool.reply_to_post(self, post_id, content, subreddit)

    @wrap_tool("agno__reddit__reply_to_comment", AgnoRedditTools.reply_to_comment)
    def reply_to_comment(
        self, comment_id: str, content: str, subreddit: Optional[str] = None
    ) -> str:
        return self._tool.reply_to_comment(self, comment_id, content, subreddit)
