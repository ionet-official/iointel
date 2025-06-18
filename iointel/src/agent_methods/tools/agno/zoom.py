from typing import Optional
from agno.tools.zoom import ZoomTools as AgnoZoomTools
from .common import make_base, wrap_tool
from pydantic import Field


class Zoom(make_base(AgnoZoomTools)):
    account_id: Optional[str] = Field(default=None, frozen=True)
    client_id: Optional[str] = Field(default=None, frozen=True)
    client_secret: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            account_id=self.account_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    @wrap_tool("agno__zoom__get_access_token", AgnoZoomTools.get_access_token)
    def get_access_token(self) -> str:
        return self._tool.get_access_token()

    @wrap_tool("agno__zoom__schedule_meeting", AgnoZoomTools.schedule_meeting)
    def schedule_meeting(
        self, topic: str, start_time: str, duration: int, timezone: str = "UTC"
    ) -> str:
        return self._tool.schedule_meeting(topic, start_time, duration, timezone)

    @wrap_tool("agno__zoom__get_upcoming_meetings", AgnoZoomTools.get_upcoming_meetings)
    def get_upcoming_meetings(self, user_id: str = "me") -> str:
        return self._tool.get_upcoming_meetings(user_id)

    @wrap_tool("agno__zoom__list_meetings", AgnoZoomTools.list_meetings)
    def list_meetings(self, user_id: str = "me", type: str = "scheduled") -> str:
        return self._tool.list_meetings(user_id, type)

    @wrap_tool(
        "agno__zoom__get_meeting_recordings", AgnoZoomTools.get_meeting_recordings
    )
    def get_meeting_recordings(
        self,
        meeting_id: str,
        include_download_token: bool = False,
        token_ttl: Optional[int] = None,
    ) -> str:
        return self.get_meeting_recordings(
            meeting_id, include_download_token, token_ttl
        )

    @wrap_tool("agno__zoom__delete_meeting", AgnoZoomTools.delete_meeting)
    def delete_meeting(
        self, meeting_id: str, schedule_for_reminder: bool = True
    ) -> str:
        return self._tool.delete_meeting(meeting_id, schedule_for_reminder)

    @wrap_tool("agno__zoom__get_meeting", AgnoZoomTools.get_meeting)
    def get_meeting(self, meeting_id: str) -> str:
        return self._tool.get_meeting(meeting_id)

    @wrap_tool("agno__zoom__instructions", AgnoZoomTools.instructions)
    def instructions(self) -> str:
        return self._tool.instructions()
