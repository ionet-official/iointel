from agno.tools.telegram import TelegramTools as AgnoTelegramTools
from .common import make_base, wrap_tool


class Telegram(make_base(AgnoTelegramTools)):
    def _get_tool(self):
        return self.Inner(
            chat_id=self.chat_id_,
            token=self.token_,
        )

    @wrap_tool("agno__telegram___call_post_method", AgnoTelegramTools._call_post_method)
    def _call_post_method(self, method, *args, **kwargs):
        return self._call_post_method(self, method, args, kwargs)

    @wrap_tool("agno__telegram__send_message", AgnoTelegramTools.send_message)
    def send_message(self, message: str) -> str:
        return self.send_message(self, message)
