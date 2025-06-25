from typing import Any, Optional, Union, Dict
from agno.tools.models_labs import ModelsLabTools as AgnoModelsLabTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team
from agno.models.response import FileType
from pydantic import Field


class ModelsLab(make_base(AgnoModelsLabTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    wait_for_completion: bool = Field(default=False, frozen=True)
    add_to_eta: int = Field(default=15, frozen=True)
    max_wait_time: int = Field(default=60, frozen=True)
    file_type: FileType = Field(default=FileType.MP4, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            wait_for_completion=self.wait_for_completion,
            add_to_eta=self.add_to_eta,
            max_wait_time=self.max_wait_time,
            file_type=self.file_type,
        )

    @wrap_tool("agno__modelslab___create_payload", AgnoModelsLabTools._create_payload)
    def _create_payload(self, prompt: str) -> Dict[str, Any]:
        return self._tool._create_payload(prompt)

    @wrap_tool("agno__modelslab__generate_media", AgnoModelsLabTools.generate_media)
    def generate_media(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self._tool.generate_media(agent, prompt)
