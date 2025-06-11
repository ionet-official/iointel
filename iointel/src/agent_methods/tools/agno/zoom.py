from typing import Optional
from agno.tools.zoom import ZoomTools as AgnoZoomTools
from .common import make_base, wrap_tool


class Zoom(make_base(AgnoZoomTools)):
    def _get_tool(self):
        return self.Inner(
            account_id=self.account_id_,
            client_id=self.client_id_,
            client_secret=self.client_secret_,
        )

    @wrap_tool("agno__zoom__get_access_token", AgnoZoomTools.get_access_token)
    def get_access_token(self) -> str:
        return self.get_access_token(self)

    @wrap_tool("agno__zoom__schedule_meeting", AgnoZoomTools.schedule_meeting)
    def schedule_meeting(
        self, topic: str, start_time: str, duration: int, timezone: str = "UTC"
    ) -> str:
        return self.schedule_meeting(self, topic, start_time, duration, timezone)

    @wrap_tool("agno__zoom__get_upcoming_meetings", AgnoZoomTools.get_upcoming_meetings)
    def get_upcoming_meetings(self, user_id: str = "me") -> str:
        return self.get_upcoming_meetings(self, user_id)

    @wrap_tool("agno__zoom__list_meetings", AgnoZoomTools.list_meetings)
    def list_meetings(self, user_id: str = "me", type: str = "scheduled") -> str:
        return self.list_meetings(self, user_id, type)

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
            self, meeting_id, include_download_token, token_ttl
        )

    @wrap_tool("agno__zoom__delete_meeting", AgnoZoomTools.delete_meeting)
    def delete_meeting(
        self, meeting_id: str, schedule_for_reminder: bool = True
    ) -> str:
        return self.delete_meeting(self, meeting_id, schedule_for_reminder)

    @wrap_tool("agno__zoom__get_meeting", AgnoZoomTools.get_meeting)
    def get_meeting(self, meeting_id: str) -> str:
        return self.get_meeting(self, meeting_id)

    @wrap_tool("agno__zoom__instructions", AgnoZoomTools.instructions)
    def instructions(self) -> str:
        return self.instructions(self)
