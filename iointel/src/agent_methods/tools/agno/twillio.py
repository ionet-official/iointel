from typing import Any, List, Dict, Optional
from agno.tools.twilio import TwilioTools as AgnoTwilioTools
from .common import make_base, wrap_tool
from pydantic import Field


class Twilio(make_base(AgnoTwilioTools)):
    account_sid: Optional[str] = Field(default=None, frozen=True)
    auth_token: Optional[str] = Field(default=None, frozen=True)
    api_key: Optional[str] = Field(default=None, frozen=True)
    api_secret: Optional[str] = Field(default=None, frozen=True)
    region: Optional[str] = Field(default=None, frozen=True)
    edge: Optional[str] = Field(default=None, frozen=True)
    debug: bool = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            account_sid=self.account_sid,
            auth_token=self.auth_token,
            api_key=self.api_key,
            api_secret=self.api_secret,
            region=self.region,
            edge=self.edge,
            debug=self.debug,
        )

    @wrap_tool(
        "agno__twilio__validate_phone_number", AgnoTwilioTools.validate_phone_number
    )
    def validate_phone_number(self, phone: str) -> bool:
        return self._tool.validate_phone_number(phone)

    @wrap_tool("agno__twilio__send_sms", AgnoTwilioTools.send_sms)
    def send_sms(self, to: str, from_: str, body: str) -> str:
        return self._tool.send_sms(to, from_, body)

    @wrap_tool("agno__twilio__get_call_details", AgnoTwilioTools.get_call_details)
    def get_call_details(self, call_sid: str) -> Dict[str, Any]:
        return self._tool.get_call_details(call_sid)

    @wrap_tool("agno__twilio__list_messages", AgnoTwilioTools.list_messages)
    def list_messages(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._tool.list_messages(limit)
