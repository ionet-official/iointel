from typing import Optional
from agno.tools.browserbase import BrowserbaseTools as AgnoBrowserbaseTools
from pydantic import Field

from .common import make_base, wrap_tool


class Browserbase(make_base(AgnoBrowserbaseTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    project_id: Optional[str] = Field(default=None, frozen=True)
    base_url: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            project_id=self.project_id,
            base_url=self.base_url,
        )

    @wrap_tool("agno__browserbase__make_request", AgnoBrowserbaseTools.navigate_to)
    def navigate_to(self, url: str, connect_url: Optional[str] = None) -> str:
        return self._tool.navigate_to(url, connect_url)

    @wrap_tool("agno__browserbase__screenshot", AgnoBrowserbaseTools.screenshot)
    def screenshot(
        self, path: str, full_page: bool = True, connect_url: Optional[str] = None
    ) -> str:
        return self._tool.screenshot(path, full_page, connect_url)

    @wrap_tool(
        "agno__browserbase__get_page_content", AgnoBrowserbaseTools.get_page_content
    )
    def get_page_content(self, connect_url: Optional[str] = None) -> str:
        return self._tool.get_page_content(connect_url)

    @wrap_tool("agno__browserbase__close_session", AgnoBrowserbaseTools.close_session)
    def close_session(self) -> str:
        return self._tool.close_session()
