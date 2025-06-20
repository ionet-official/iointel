from typing import Any, Optional, List, Dict
from agno.tools.whatsapp import WhatsAppTools as AgnoWhatsAppTools
from .common import make_base, wrap_tool
from pydantic import Field


class WhatsApp(make_base(AgnoWhatsAppTools)):
    access_token: Optional[str] = Field(default=None, frozen=True)
    phone_number_id: Optional[str] = Field(default=None, frozen=True)
    version: str = Field(default="v22.0", frozen=True)
    recipient_waid: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            access_token=self.access_token,
            phone_number_id=self.phone_number_id,
            version=self.version,
            recipient_waid=self.recipient_waid,
            async_mode=True,
        )

    @wrap_tool(
        "agno__whatsapp__send_text_message_sync",
        AgnoWhatsAppTools.send_text_message_sync,
    )
    def send_text_message_sync(
        self,
        text: str = "",
        recipient: Optional[str] = None,
        preview_url: bool = False,
        recipient_type: str = "individual",
    ) -> str:
        return self._tool.send_text_message_sync(
            text, recipient, preview_url, recipient_type
        )

    @wrap_tool(
        "agno__whatsapp__send_template_message_sync",
        AgnoWhatsAppTools.send_template_message_sync,
    )
    def send_template_message_sync(
        self,
        recipient: Optional[str] = None,
        template_name: str = "",
        language_code: str = "en_US",
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self._tool.send_template_message_sync(
            recipient, template_name, language_code, components
        )

    @wrap_tool(
        "agno__whatsapp__send_text_message_async",
        AgnoWhatsAppTools.send_text_message_async,
    )
    def send_text_message_async(
        self,
        text: str = "",
        recipient: Optional[str] = None,
        preview_url: bool = False,
        recipient_type: str = "individual",
    ) -> str:
        return self._tool.send_text_message_async(
            text, recipient, preview_url, recipient_type
        )

    @wrap_tool(
        "agno__whatsapp__send_template_message_async",
        AgnoWhatsAppTools.send_template_message_async,
    )
    def send_template_message_async(
        self,
        recipient: Optional[str] = None,
        template_name: str = "",
        language_code: str = "en_US",
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self._tool.send_template_message_async(
            recipient, template_name, language_code, components
        )
