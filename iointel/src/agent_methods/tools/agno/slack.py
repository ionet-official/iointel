from typing import Optional
from agno.tools.slack import SlackTools as AgnoSlackTools
from .common import make_base, wrap_tool
from pydantic import Field


class Slack(make_base(AgnoSlackTools)):
    token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            token=self.token,
            send_message=True,
            list_channels=True,
            get_channel_history=True,
        )

    @wrap_tool("agno__slack__send_message", AgnoSlackTools.send_message)
    def send_message(self, channel: str, text: str) -> str:
        return self.send_message(channel, text)

    @wrap_tool("agno__slack__list_channels", AgnoSlackTools.list_channels)
    def list_channels(self) -> str:
        return self.list_channels()

    @wrap_tool("agno__slack__get_channel_history", AgnoSlackTools.get_channel_history)
    def get_channel_history(self, channel: str, limit: int = 100) -> str:
        return self.get_channel_history(channel, limit)
