from typing import Optional
from agno.tools.x import XTools as AgnoXTools
from .common import make_base, wrap_tool
from pydantic import Field


class X(make_base(AgnoXTools)):
    bearer_token: Optional[str] = Field(default=None, frozen=True)
    consumer_key: Optional[str] = Field(default=None, frozen=True)
    consumer_secret: Optional[str] = Field(default=None, frozen=True)
    access_token: Optional[str] = Field(default=None, frozen=True)
    access_token_secret: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            bearer_token=self.bearer_token,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_tokens,
            access_token_secret=self.access_token_secret,
        )

    @wrap_tool("agno__x__create_post", AgnoXTools.create_post)
    def create_post(self, text: str) -> str:
        return self._tool.create_post(text)

    @wrap_tool("agno__x__reply_to_post", AgnoXTools.reply_to_post)
    def reply_to_post(self, post_id: str, text: str) -> str:
        return self._tool.reply_to_post(post_id, text)

    @wrap_tool("agno__x__send_dm", AgnoXTools.send_dm)
    def send_dm(self, recipient: str, text: str) -> str:
        return self._tool.send_dm(recipient, text)

    @wrap_tool("agno__x__get_my_info", AgnoXTools.get_my_info)
    def get_my_info(self) -> str:
        return self._tool.get_my_info(self)

    @wrap_tool("agno__x__get_user_info", AgnoXTools.get_user_info)
    def get_user_info(self, username: str) -> str:
        return self._tool.get_user_info(username)

    @wrap_tool("agno__x__get_home_timeline", AgnoXTools.get_home_timeline)
    def get_home_timeline(self, max_results: int = 10) -> str:
        return self._tool.get_home_timeline(max_results)
