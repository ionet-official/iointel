from typing import Optional
from agno.tools.webex import WebexTools as AgnoWebexTools
from .common import make_base, wrap_tool


class Webex(make_base(AgnoWebexTools)):
    access_token: Optional[str] = None

    def _get_tool(self):
        return self.Inner(
            send_message=True,
            list_rooms=True,
            access_token=self.access_token,
        )

    @wrap_tool("agno__webex__send_message", AgnoWebexTools.send_message)
    def send_message(self, room_id: str, text: str) -> str:
        return self.send_message(room_id, text)

    @wrap_tool("agno__webex__list_rooms", AgnoWebexTools.list_rooms)
    def list_rooms(self) -> str:
        return self.list_rooms()
