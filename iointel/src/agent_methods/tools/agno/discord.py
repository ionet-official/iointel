from typing import Optional
from agno.tools.discord import DiscordTools as AgnoDiscordTools
from pydantic import Field
from .common import make_base, wrap_tool


class Discord(make_base(AgnoDiscordTools)):
    bot_token: Optional[str] = (None,)
    enable_messaging: bool = Field(required=True, frozen=True)
    enable_history: bool = Field(required=True, frozen=True)
    enable_channel_management: bool = Field(required=True, frozen=True)
    enable_message_management: bool = Field(required=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            bot_token=self.bot_token,
            enable_messaging=self.enable_messaging,
            enable_history=self.enable_history,
            enable_channel_management=self.enable_channel_management,
            enable_message_management=self.enable_message_management,
        )

    @wrap_tool("agno__discord__send_message", AgnoDiscordTools.send_message)
    def send_message(self, channel_id: int, message: str) -> str:
        return self._tool.send_message(channel_id, message)

    @wrap_tool("agno__discord__get_channel_info", AgnoDiscordTools.get_channel_info)
    def get_channel_info(self, channel_id: int) -> str:
        return self._tool.get_channel_info(channel_id)

    @wrap_tool("agno__discord__list_channels", AgnoDiscordTools.list_channels)
    def list_channels(self, guild_id: int) -> str:
        return self._tool.list_channels(guild_id)

    @wrap_tool(
        "agno__discord__list_get_channel_messages",
        AgnoDiscordTools.get_channel_messages,
    )
    def get_channel_messages(self, channel_id: int, limit: int = 100) -> str:
        return self._tool.get_channel_messages(channel_id, limit)

    @wrap_tool("agno__discord__delete_message", AgnoDiscordTools.delete_message)
    def delete_message(self, channel_id: int, message_id: int) -> str:
        return self._tool.delete_message(channel_id, message_id)
