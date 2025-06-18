from typing import Any, Optional, List, Dict
from agno.tools.whatsapp import WhatsAppTools as AgnoWhatsAppTools
from .common import make_base, wrap_tool


class WhatsApp(make_base(AgnoWhatsAppTools)):
    def _get_tool(self):
        return self.Inner(
            access_token=self.access_token_,
            phone_number_id=self.phone_number_id_,
            version=self.version_,
            recipient_waid=self.recipient_waid_,
            async_mode=self.async_mode_,
        )

    @wrap_tool("agno__whatsapp___get_headers", AgnoWhatsAppTools._get_headers)
    def _get_headers(self) -> Dict[str, str]:
        return self._get_headers(self)

    @wrap_tool("agno__whatsapp___get_messages_url", AgnoWhatsAppTools._get_messages_url)
    def _get_messages_url(self) -> str:
        return self._get_messages_url()

    @wrap_tool(
        "agno__whatsapp___send_message_async", AgnoWhatsAppTools._send_message_async
    )
    def _send_message_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._send_message_async(data)

    @wrap_tool(
        "agno__whatsapp___send_message_sync", AgnoWhatsAppTools._send_message_sync
    )
    def _send_message_sync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._send_message_sync(data)

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
        return self.send_text_message_sync(
            self, text, recipient, preview_url, recipient_type
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
        return self.send_template_message_sync(
            self, recipient, template_name, language_code, components
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
        return self.send_text_message_async(
            self, text, recipient, preview_url, recipient_type
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
        return self.send_template_message_async(
            self, recipient, template_name, language_code, components
        )
