from typing import Any, Optional, Union, Dict
from agno.tools.models_labs import ModelsLabTools as AgnoModelsLabTools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.team.team import Team


class ModelsLab(make_base(AgnoModelsLabTools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            wait_for_completion=self.wait_for_completion_,
            add_to_eta=self.add_to_eta_,
            max_wait_time=self.max_wait_time_,
            file_type=self.file_type_,
        )

    @wrap_tool("agno__modelslab___create_payload", AgnoModelsLabTools._create_payload)
    def _create_payload(self, prompt: str) -> Dict[str, Any]:
        return self._create_payload(self, prompt)

    @wrap_tool(
        "agno__modelslab___add_media_artifact", AgnoModelsLabTools._add_media_artifact
    )
    def _add_media_artifact(
        self,
        agent: Union[Agent, Team],
        media_id: str,
        media_url: str,
        eta: Optional[str] = None,
    ) -> None:
        return self._add_media_artifact(self, agent, media_id, media_url, eta)

    @wrap_tool("agno__modelslab___wait_for_media", AgnoModelsLabTools._wait_for_media)
    def _wait_for_media(self, media_id: str, eta: int) -> bool:
        return self._wait_for_media(self, media_id, eta)

    @wrap_tool("agno__modelslab__generate_media", AgnoModelsLabTools.generate_media)
    def generate_media(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self.generate_media(self, agent, prompt)
