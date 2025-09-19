from typing import Optional, List
from agno.tools.googlecalendar import GoogleCalendarTools as AgnoGoogleCalendarTools
from .common import make_base, wrap_tool
from pydantic import Field


class GoogleCalendar(make_base(AgnoGoogleCalendarTools)):
    credentials_path: Optional[str] = Field(default=None, frozen=True)
    token_path: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            credentials_path=self.credentials_path, token_path=self.token_path
        )

    @wrap_tool("google_calendar_list_events", AgnoGoogleCalendarTools.list_events)
    def list_events(self, limit: int = 10, date_from: str = None) -> str:
        return self._tool.list_events(limit=limit, date_from=date_from)

    @wrap_tool("google_calendar_create_event", AgnoGoogleCalendarTools.create_event)
    def create_event(
        self,
        start_datetime: str,
        end_datetime: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        timezone: Optional[str] = None,
        attendees: List[str] = [],
        send_updates: Optional[str] = "all",
        add_google_meet_link: Optional[bool] = False,
    ) -> str:
        return self._tool.create_event(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            title=title,
            description=description,
            location=location,
            timezone=timezone,
            attendees=attendees,
            send_updates=send_updates,
            add_google_meet_link=add_google_meet_link,
        )
