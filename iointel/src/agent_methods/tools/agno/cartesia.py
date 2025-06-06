from typing import Optional, Union
from agno.tools.cartesia import CartesiaTools as AgnoCartesiaTools
from agno.team.team import Team
from iointel.src.agents import Agent

from .common import make_base, wrap_tool


class Cartesia(make_base(AgnoCartesiaTools)):
    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("agno__cartesia__list_voices", AgnoCartesiaTools.list_voices)
    def list_voices(self) -> str:
        return self._tool.list_voices()

    @wrap_tool("agno__cartesia__localize_voice", AgnoCartesiaTools.localize_voice)
    def localize_voice(
        self,
        name: str,
        description: str,
        language: str,
        original_speaker_gender: str,
        voice_id: Optional[str] = None,
    ) -> str:
        return self._tool.localize_voice(
            name,
            description,
            language,
            original_speaker_gender,
            voice_id,
        )

    @wrap_tool("agno__cartesia__text_to_speech", AgnoCartesiaTools.text_to_speech)
    def text_to_speech(
        self,
        agent: Union[Agent, Team],
        transcript: str,
        voice_id: Optional[str] = None,
    ) -> str:
        return self._tool.text_to_speech(agent, transcript, voice_id)
