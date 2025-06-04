from functools import wraps
from typing import Optional
from pydantic import BaseModel, Field
from agno.tools.email import EmailTools as AgnoEmailTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class Email(BaseModel, DisableAgnoRegistryMixin, AgnoEmailTools):
    receiver_email: Optional[str] = Field(default=None)
    sender_name: Optional[str] = Field(default=None)
    sender_email: Optional[str] = Field(default=None)
    sender_passkey: Optional[str] = Field(default=None)

    def __init__(
        self,
        receiver_email: Optional[str] = None,
        sender_name: Optional[str] = None,
        sender_email: Optional[str] = None,
        sender_passkey: Optional[str] = None,
    ):
        super().__init__(
            receiver_email=receiver_email,
            sender_name=sender_name,
            sender_email=sender_email,
            sender_passkey=sender_passkey,
        )

    @register_tool(name="email_user")
    @wraps(AgnoEmailTools.email_user)
    def email_user(self, subject: str, body: str) -> str:
        """
        Send an email message to the user.

        Args:
            message: The message content to send
        """
        return super().email_user(subject=subject, body=body)
