from typing import Optional, Union
from agno.tools.desi_vocal import DesiVocalTools as AgnoDesiVocalTools
from agno.agent import Agent
from agno.team.team import Team
from pydantic import Field

from .common import make_base, wrap_tool


class DesiVocal(make_base(AgnoDesiVocalTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    voice_id: Optional[str] = Field(
        default="f27d74e5-ea71-4697-be3e-f04bbd80c1a8", frozen=True
    )

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            voice_id=self.voice_id,
        )

    @wrap_tool("agno__desivocal__get_voices", AgnoDesiVocalTools.get_voices)
    def get_voices(self) -> str:
        return self._tool.get_voices()

    @wrap_tool("agno__desivocal__text_to_speech", AgnoDesiVocalTools.text_to_speech)
    def text_to_speech(
        self, agent: Union[Agent, Team], prompt: str, voice_id: Optional[str] = None
    ) -> str:
        return self._tool.get_voices(agent, prompt, voice_id)
