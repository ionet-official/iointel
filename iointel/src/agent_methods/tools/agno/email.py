from agno.tools.email import EmailTools as AgnoEmailTools

from .common import make_base, wrap_tool


class Email(make_base(AgnoEmailTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("email_user", AgnoEmailTools.email_user)
    def email_user(self, subject: str, body: str) -> str:
        """
        Send an email message to the user.

        Args:
            message: The message content to send
        """
        return super().email_user(subject=subject, body=body)
