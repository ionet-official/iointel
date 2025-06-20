from typing import List, Optional
from agno.tools.gmail import GmailTools as AgnoGmailTools
from google.oauth2.credentials import Credentials
from .common import make_base, wrap_tool
from pydantic import Field


class Gmail(make_base(AgnoGmailTools)):
    get_latest_emails_enabled: bool = Field(default=True, frozen=True)
    get_emails_from_user_enabled: bool = Field(default=True, frozen=True)
    get_unread_emails_enabled: bool = Field(default=True, frozen=True)
    get_starred_emails_enabled: bool = Field(default=True, frozen=True)
    get_emails_by_context_enabled: bool = Field(default=True, frozen=True)
    get_emails_by_date_enabled: bool = Field(default=True, frozen=True)
    get_emails_by_thread_enabled: bool = Field(default=True, frozen=True)
    create_draft_email_enabled: bool = Field(default=True, frozen=True)
    send_email_enabled: bool = Field(default=True, frozen=True)
    send_email_reply_enabled: bool = Field(default=True, frozen=True)
    search_emails_enabled: bool = Field(default=True, frozen=True)
    creds_: Optional[Credentials] = Field(default=None, frozen=True)
    credentials_path_: Optional[str] = Field(default=None, frozen=True)
    token_path_: Optional[str] = Field(default=None, frozen=True)
    scopes_: Optional[List[str]] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            get_latest_emails=self.get_latest_emails_enabled,
            get_emails_from_user=self.get_emails_from_user_enabled,
            get_unread_emails=self.get_unread_emails_enabled,
            get_starred_emails=self.get_starred_emails_enabled,
            get_emails_by_context=self.get_emails_by_context_enabled,
            get_emails_by_date=self.get_emails_by_date_enabled,
            get_emails_by_thread=self.get_emails_by_thread_enabled,
            create_draft_email=self.create_draft_email_enabled,
            send_email=self.send_email_enabled,
            send_email_reply=self.send_email_reply_enabled,
            search_emails=self.search_emails_enabled,
            creds=self.creds_,
            credentials_path=self.credentials_path_,
            token_path=self.token_path_,
            scopes=self.scopes_,
        )

    @wrap_tool("agno__gmail__get_latest_emails", AgnoGmailTools.get_latest_emails)
    def get_latest_emails(self, count: int) -> str:
        return self._tool.get_latest_emails(count)

    @wrap_tool("agno__gmail__get_emails_from_user", AgnoGmailTools.get_emails_from_user)
    def get_emails_from_user(self, user: str, count: int) -> str:
        return self._tool.get_emails_from_user(user, count)

    @wrap_tool("agno__gmail__get_unread_emails", AgnoGmailTools.get_unread_emails)
    def get_unread_emails(self, count: int) -> str:
        return self._tool.get_unread_emails(count)

    @wrap_tool("agno__gmail__get_emails_by_thread", AgnoGmailTools.get_emails_by_thread)
    def get_emails_by_thread(self, thread_id: str) -> str:
        return self._tool.get_emails_by_thread(thread_id)

    @wrap_tool("agno__gmail__get_starred_emails", AgnoGmailTools.get_starred_emails)
    def get_starred_emails(self, count: int) -> str:
        return self._tool.get_starred_emails(count)

    @wrap_tool(
        "agno__gmail__get_emails_by_context", AgnoGmailTools.get_emails_by_context
    )
    def get_emails_by_context(self, context: str, count: int) -> str:
        return self._tool.get_emails_by_context(context, count)

    @wrap_tool("agno__gmail__get_emails_by_date", AgnoGmailTools.get_emails_by_date)
    def get_emails_by_date(
        self,
        start_date: int,
        range_in_days: Optional[int] = None,
        num_emails: Optional[int] = 10,
    ) -> str:
        return self._tool.get_emails_by_date(start_date, range_in_days, num_emails)

    @wrap_tool("agno__gmail__create_draft_email", AgnoGmailTools.create_draft_email)
    def create_draft_email(
        self, to: str, subject: str, body: str, cc: Optional[str] = None
    ) -> str:
        return self._tool.create_draft_email(to, subject, body, cc)

    @wrap_tool("agno__gmail__send_email", AgnoGmailTools.send_email)
    def send_email(
        self, to: str, subject: str, body: str, cc: Optional[str] = None
    ) -> str:
        return self._tool.send_email(to, subject, body, cc)

    @wrap_tool("agno__gmail__send_email_reply", AgnoGmailTools.send_email_reply)
    def send_email_reply(
        self,
        thread_id: str,
        message_id: str,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
    ) -> str:
        return self._tool.send_email_reply(thread_id, message_id, to, subject, body, cc)

    @wrap_tool("agno__gmail__search_emails", AgnoGmailTools.search_emails)
    def search_emails(self, query: str, count: int) -> str:
        return self._tool.search_emails(query, count)
