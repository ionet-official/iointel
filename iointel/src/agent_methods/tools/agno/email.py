from typing import Optional
from agno.tools.email import EmailTools as AgnoEmailTools
from pydantic import Field

from .common import make_base, wrap_tool


class Email(make_base(AgnoEmailTools)):
    receiver_email: Optional[str] = Field(default=None, frozen=True)
    sender_name: Optional[str] = Field(default=None, frozen=True)
    sender_email: Optional[str] = Field(default=None, frozen=True)
    sender_passkey: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            receiver_email=self.receiver_email,
            sender_name=self.sender_name,
            sender_email=self.sender_email,
            sender_passkey=self.sender_passkey,
        )

    @wrap_tool("email_user", AgnoEmailTools.email_user)
    def email_user(self, subject: str, body: str) -> str:
        return super().email_user(subject=subject, body=body)
