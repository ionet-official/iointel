from typing import Optional, Union
from agno.tools.reasoning import ReasoningTools as AgnoReasoningTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team


class Reasoning(make_base(AgnoReasoningTools)):
    def _get_tool(self):
        return self.Inner(
            think=self.think_,
            analyze=self.analyze,
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
            add_few_shot=self.add_few_shot_,
            few_shot_examples=self.few_shot_examples_,
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
        return self.think(self, agent, title, thought, action, confidence)

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
        return self.analyze(
            self, agent, title, result, analysis, next_action, confidence
        )
