from typing import Optional
from agno.tools.wikipedia import WikipediaTools as AgnoWikipediaTools
from .common import make_base, wrap_tool
from agno.knowledge.wikipedia import WikipediaKnowledgeBase
from pydantic import Field


class Wikipedia(make_base(AgnoWikipediaTools)):
    knowledge_base: Optional[WikipediaKnowledgeBase] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            knowledge_base=self.knowledge_base,
        )

    @wrap_tool(
        "agno__wikipedia__search_wikipedia_and_update_knowledge_base",
        AgnoWikipediaTools.search_wikipedia_and_update_knowledge_base,
    )
    def search_wikipedia_and_update_knowledge_base(self, topic: str) -> str:
        return self.search_wikipedia_and_update_knowledge_base(topic)

    @wrap_tool("agno__wikipedia__search_wikipedia", AgnoWikipediaTools.search_wikipedia)
    def search_wikipedia(self, query: str) -> str:
        return self.search_wikipedia(query)
