from typing import Optional, Union
from agno.tools.fal import FalTools as AgnoFalTools
from agno.agent import Agent
from agno.team import Team
from .common import make_base, wrap_tool
from pydantic import Field


class Fal(make_base(AgnoFalTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    model: str = Field(default="fal-ai/hunyuan-video", frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            model=self.model,
        )

    @wrap_tool("agno__fal__on_queue_update", AgnoFalTools.on_queue_update)
    def on_queue_update(self, update):
        return self._tool.on_queue_update()

    @wrap_tool("agno__fal__generate_media", AgnoFalTools.generate_media)
    def generate_media(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self._tool.generate_media(agent, prompt)

    @wrap_tool("agno__fal__image_to_image", AgnoFalTools.image_to_image)
    def image_to_image(
        self, agent: Union[Agent, Team], prompt: str, image_url: Optional[str] = None
    ) -> str:
        return self._tool.image_to_image(agent, prompt, image_url)
