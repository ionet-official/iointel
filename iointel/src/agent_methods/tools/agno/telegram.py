from typing import Optional, Union
from agno.tools.telegram import TelegramTools as AgnoTelegramTools
from .common import make_base, wrap_tool
from pydantic import Field


class Telegram(make_base(AgnoTelegramTools)):
    chat_id: Union[str, int] = Field(frozen=True)
    token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            chat_id=self.chat_id,
            token=self.token,
        )

    @wrap_tool("agno__telegram__send_message", AgnoTelegramTools.send_message)
    def send_message(self, message: str) -> str:
        return self._tool.send_message(self, message)
