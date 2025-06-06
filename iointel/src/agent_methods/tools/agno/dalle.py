from typing import Union
from agno.tools.dalle import DalleTools as AgnoDalleTools
from agno.team.team import Team
from iointel.src.agents import Agent

from .common import make_base, wrap_tool


class Dalle(make_base(AgnoDalleTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("agno__dalle__create_image", AgnoDalleTools.create_image)
    def create_image(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self._tool.create_image(agent, prompt)
