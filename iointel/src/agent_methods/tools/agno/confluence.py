from typing import Optional
from agno.tools.confluence import ConfluenceTools as AgnoConfluenceTools

from .common import make_base, wrap_tool


class Confluence(make_base(AgnoConfluenceTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool(
        "agno__confluence__get_page_content", AgnoConfluenceTools.get_page_content
    )
    def get_page_content(
        self, space_name: str, page_title: str, expand: Optional[str] = "body.storage"
    ) -> str:
        return self._tool.get_page_content(space_name, page_title, expand)

    @wrap_tool(
        "agno__confluence__get_all_space_detail",
        AgnoConfluenceTools.get_all_space_detail,
    )
    def get_all_space_detail(self) -> str:
        return self._tool.get_all_space_detail()

    @wrap_tool("agno__confluence__get_space_key", AgnoConfluenceTools.get_space_key)
    def get_space_key(self, space_name: str) -> str:
        return self._tool.get_space_key(space_name)

    @wrap_tool(
        "agno__confluence__get_all_page_from_space",
        AgnoConfluenceTools.get_all_page_from_space,
    )
    def get_all_page_from_space(self, space_name: str) -> str:
        return self._tool.get_all_page_from_space(space_name)

    @wrap_tool("agno__confluence__create_page", AgnoConfluenceTools.create_page)
    def create_page(
        self, space_name: str, title: str, body: str, parent_id: Optional[str] = None
    ) -> str:
        return self._tool.create_page(space_name, title, body, parent_id)

    @wrap_tool("agno__confluence__update_page", AgnoConfluenceTools.update_page)
    def update_page(self, page_id: str, title: str, body: str) -> str:
        return self._tool.update_page(page_id, title, body)
