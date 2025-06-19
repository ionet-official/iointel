from typing import Optional, Union
from agno.tools.reasoning import ReasoningTools as AgnoReasoningTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team
from pydantic import Field


class Reasoning(make_base(AgnoReasoningTools)):
    instructions: Optional[str] = Field(default=None, frozen=True)
    add_instructions: bool = Field(default=False, frozen=True)
    add_few_shot: bool = Field(default=False, frozen=True)
    few_shot_examples: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            think=True,
            analyze=True,
            instructions=self.instructions,
            add_instructions=self.add_instructions,
            add_few_shot=self.add_few_shot,
            few_shot_examples=self.few_shot_examples,
        )

    @wrap_tool("agno__reasoning__think", AgnoReasoningTools.think)
    def think(
        self,
        agent: Union[Agent, Team],
        title: str,
        thought: str,
        action: Optional[str] = None,
        confidence: float = 0.8,
    ) -> str:
        return self._tool.think(agent, title, thought, action, confidence)

    @wrap_tool("agno__reasoning__analyze", AgnoReasoningTools.analyze)
    def analyze(
        self,
        agent: Union[Agent, Team],
        title: str,
        result: str,
        analysis: str,
        next_action: str = "continue",
        confidence: float = 0.8,
    ) -> str:
        return self._tool.analyze(
            agent, title, result, analysis, next_action, confidence
        )
