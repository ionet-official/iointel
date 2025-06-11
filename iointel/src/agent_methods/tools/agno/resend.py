from agno.tools.resend import ResendTools as AgnoResendTools
from .common import make_base, wrap_tool


class Resend(make_base(AgnoResendTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            from_email=self.from_email_,
        )

    @wrap_tool("agno__resend__send_email", AgnoResendTools.send_email)
    def send_email(self, to_email: str, subject: str, body: str) -> str:
        return self.send_email(self, to_email, subject, body)
