from typing import Union
from agno.tools.thinking import ThinkingTools as AgnoThinkingTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team


class Thinking(make_base(AgnoThinkingTools)):
    def _get_tool(self):
        return self.Inner(
            think=self.think,
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
        )

    @wrap_tool("agno__thinking__think", AgnoThinkingTools.think)
    def think(self, agent: Union[Agent, Team], thought: str) -> str:
        return self.think(self, agent, thought)
