from agno.tools.telegram import TelegramTools as AgnoTelegramTools
from .common import make_base, wrap_tool


class Telegram(make_base(AgnoTelegramTools)):
    def _get_tool(self):
        return self.Inner(
            chat_id=self.chat_id_,
            token=self.token_,
        )

    @wrap_tool("agno__telegram__send_message", AgnoTelegramTools.send_message)
    def send_message(self, message: str) -> str:
        return self._tool.send_message(self, message)
