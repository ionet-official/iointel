from typing import Union
from agno.tools.knowledge import KnowledgeTools as AgnoKnowledgeTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team


class Knowledge(make_base(AgnoKnowledgeTools)):
    def _get_tool(self):
        return self.Inner(
            knowledge=self.knowledge_,
            think=self.think_,
            search=self.search_,
            analyze=self.analyze,
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
            add_few_shot=self.add_few_shot_,
            few_shot_examples=self.few_shot_examples_,
        )

    @wrap_tool("agno__knowledge__think", AgnoKnowledgeTools.think)
    def think(self, agent: Union[Agent, Team], thought: str) -> str:
        return self.think(self, agent, thought)

    @wrap_tool("agno__knowledge__search", AgnoKnowledgeTools.search)
    def search(self, agent: Union[Agent, Team], query: str) -> str:
        return self.search(self, agent, query)

    @wrap_tool("agno__knowledge__analyze", AgnoKnowledgeTools.analyze)
    def analyze(self, agent: Union[Agent, Team], analysis: str) -> str:
        return self.analyze(self, agent, analysis)
