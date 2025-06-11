from typing import Literal, Optional, Dict
from agno.tools.lumalab import LumaLabTools as AgnoLumaLabTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent


class LumaLab(make_base(AgnoLumaLabTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            wait_for_completion=self.wait_for_completion_,
            poll_interval=self.poll_interval_,
            max_wait_time=self.max_wait_time_,
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
            self, agent, prompt, start_image_url, end_image_url, loop, aspect_ratio
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
        return self.generate_video(self, agent, prompt, loop, aspect_ratio, keyframes)
