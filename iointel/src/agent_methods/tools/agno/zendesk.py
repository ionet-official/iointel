from typing import Optional
from agno.tools.zendesk import ZendeskTools as AgnoZendeskTools
from .common import make_base, wrap_tool


class Zendesk(make_base(AgnoZendeskTools)):
    username: Optional[str] = (None,)
    password: Optional[str] = (None,)
    company_name: Optional[str] = (None,)

    def _get_tool(self):
        return self.Inner(
            username=self.username_,
            password=self.password_,
            company_name=self.company_name_,
        )

    @wrap_tool("agno__zendesk__search_zendesk", AgnoZendeskTools.search_zendesk)
    def search_zendesk(self, search_string: str) -> str:
        return self._tool.search_zendesk(search_string)
