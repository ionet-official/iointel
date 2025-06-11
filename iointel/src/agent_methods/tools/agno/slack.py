from agno.tools.slack import SlackTools as AgnoSlackTools
from .common import make_base, wrap_tool


class Slack(make_base(AgnoSlackTools)):
    def _get_tool(self):
        return self.Inner(
            token=self.token_,
            send_message=self.send_message_,
            list_channels=self.list_channels_,
            get_channel_history=self.get_channel_history,
        )

    @wrap_tool("agno__slack__send_message", AgnoSlackTools.send_message)
    def send_message(self, channel: str, text: str) -> str:
        return self.send_message(self, channel, text)

    @wrap_tool("agno__slack__list_channels", AgnoSlackTools.list_channels)
    def list_channels(self) -> str:
        return self.list_channels(self)

    @wrap_tool("agno__slack__get_channel_history", AgnoSlackTools.get_channel_history)
    def get_channel_history(self, channel: str, limit: int = 100) -> str:
        return self.get_channel_history(self, channel, limit)
