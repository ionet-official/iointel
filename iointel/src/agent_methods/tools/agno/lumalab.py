from typing import Literal, Optional, Dict
from agno.tools.lumalab import LumaLabTools as AgnoLumaLabTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from pydantic import Field


class LumaLab(make_base(AgnoLumaLabTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    wait_for_completion: bool = Field(default=True, frozen=True)
    poll_interval: int = Field(default=3, frozen=True)
    max_wait_time: int = Field(default=300, frozen=True)  # 5 minutes

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            wait_for_completion=self.wait_for_completion,
            poll_interval=self.poll_interval,
            max_wait_time=self.max_wait_time,
        )

    @wrap_tool("agno__lumalab__image_to_video", AgnoLumaLabTools.image_to_video)
    def image_to_video(
        self,
        agent: Agent,
        prompt: str,
        start_image_url: str,
        end_image_url: Optional[str] = None,
        loop: bool = False,
        aspect_ratio: Literal[
            "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        ] = "16:9",
    ) -> str:
        return self.image_to_video(
            agent, prompt, start_image_url, end_image_url, loop, aspect_ratio
        )

    @wrap_tool("agno__lumalab__generate_video", AgnoLumaLabTools.generate_video)
    def generate_video(
        self,
        agent: Agent,
        prompt: str,
        loop: bool = False,
        aspect_ratio: Literal[
            "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        ] = "16:9",
        keyframes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        return self.generate_video(agent, prompt, loop, aspect_ratio, keyframes)
