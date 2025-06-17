from typing import Optional
from agno.tools.discord import DiscordTools as AgnoDiscordTools
from .common import make_base, wrap_tool


class Discord(make_base(AgnoDiscordTools)):
    bot_token: Optional[str] = (None,)

    def _get_tool(self):
        return self.Inner(
            bot_token=self.bot_token,
            enable_messaging=True,
            enable_history=True,
            enable_channel_management=True,
            enable_message_management=True,
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
        "agno__discord__get_channel_messages",
        AgnoDiscordTools.get_channel_messages,
    )
    def get_channel_messages(self, channel_id: int, limit: int = 100) -> str:
        return self._tool.get_channel_messages(channel_id, limit)

    @wrap_tool("agno__discord__delete_message", AgnoDiscordTools.delete_message)
    def delete_message(self, channel_id: int, message_id: int) -> str:
        return self._tool.delete_message(channel_id, message_id)
