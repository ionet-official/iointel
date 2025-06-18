from typing import Optional, Union
from agno.tools.website import WebsiteTools as AgnoWebsiteTools
from .common import make_base, wrap_tool

from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from pydantic import Field


class Website(make_base(AgnoWebsiteTools)):
    knowledge_base: Optional[Union[WebsiteKnowledgeBase, CombinedKnowledgeBase]] = (
        Field(default=None, frozen=True)
    )

    def _get_tool(self):
        return self.Inner(
            knowledge_base=self.knowledge_base,
        )

    @wrap_tool(
        "agno__website__add_website_to_knowledge_base",
        AgnoWebsiteTools.add_website_to_knowledge_base,
    )
    def add_website_to_knowledge_base(self, url: str) -> str:
        return self.add_website_to_knowledge_base(url)

    @wrap_tool(
        "agno__website__add_website_to_combined_knowledge_base",
        AgnoWebsiteTools.add_website_to_combined_knowledge_base,
    )
    def add_website_to_combined_knowledge_base(self, url: str) -> str:
        return self.add_website_to_combined_knowledge_base(url)

    @wrap_tool("agno__website__read_url", AgnoWebsiteTools.read_url)
    def read_url(self, url: str) -> str:
        return self.read_url(url)
