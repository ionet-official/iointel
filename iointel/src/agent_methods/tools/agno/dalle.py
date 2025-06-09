from typing import Literal, Optional, Union
from agno.tools.dalle import DalleTools as AgnoDalleTools
from agno.team.team import Team
from iointel.src.agents import Agent
from pydantic import Field
from .common import make_base, wrap_tool


class Dalle(make_base(AgnoDalleTools)):
    model: str = Field(default="dall-e-3", frozen=True)
    n: int = Field(default=1, frozen=True)
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = Field(default="1024x1024", frozen=True)
    quality: Literal["standard", "hd"] = Field(default="standard", frozen=True)
    style: Literal["vivid", "natural"] = Field(default="vivid", frozen=True)
    api_key: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            model=self.model,
            n=self.n,
            size=self.size,
            quality=self.quality,
            style=self.style,
            api_key=self.api_key,
        )

    @wrap_tool("agno__dalle__create_image", AgnoDalleTools.create_image)
    def create_image(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self._tool.create_image(agent, prompt)
