from typing import Optional, Union
from agno.tools.cartesia import CartesiaTools as AgnoCartesiaTools
from agno.team.team import Team
from iointel.src.agents import Agent
from pydantic import Field
from .common import make_base, wrap_tool


class Cartesia(make_base(AgnoCartesiaTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    model_id: str = Field(default="sonic-2", frozen=True)
    default_voice_id: str = Field(
        default="78ab82d5-25be-4f7d-82b3-7ad64e5b85b2", frozen=True
    )
    text_to_speech_enabled: bool = Field(default=True, frozen=True)
    list_voices_enabled: bool = Field(default=True, frozen=True)
    voice_localize_enabled: bool = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            model_id=self.model_id,
            default_voice_id=self.default_voice_id,
            text_to_speech_enabled=self.text_to_speech_enabled,
            list_voices_enabled=self.list_voices_enabled,
            voice_localize_enabled=self.voice_localize_enabled,
        )

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
