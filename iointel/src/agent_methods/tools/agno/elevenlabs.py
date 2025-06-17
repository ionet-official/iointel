# from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools
from typing import Optional, Union
from agno.tools.eleven_labs import ElevenLabsTools as AgnoElevenLabsTools
from agno.tools.eleven_labs import ElevenLabsAudioOutputFormat
from agno.agent import Agent
from agno.team import Team
from .common import make_base, wrap_tool
from pydantic import Field


class ElevenLabs(make_base(AgnoElevenLabsTools)):
    voice_id: str = Field(default="JBFqnCBsd6RMkjVDRZzb", fronzen=True)
    api_key: Optional[str] = Field(default=None, fronzen=True)
    target_directory: Optional[str] = Field(default=None, fronzen=True)
    model_id: str = Field(default="eleven_multilingual_v2", fronzen=True)
    output_format: ElevenLabsAudioOutputFormat = Field(
        default="mp3_44100_64", fronzen=True
    )

    def _get_tool(self):
        return self.Inner(
            voice_id=self.voice_id,
            api_key=self.api_key,
            target_directory=self.target_directory,
            model_id=self.model_id,
            output_format=self.output_format,
        )

    @wrap_tool("agno_elevenlabs_get_voices", AgnoElevenLabsTools.get_voices)
    def get_voices(self) -> str:
        return self._tool.get_voices()

    @wrap_tool(
        "agno_elevenlabs_generate_sound_effect",
        AgnoElevenLabsTools.generate_sound_effect,
    )
    def generate_sound_effect(
        self,
        agent: Union[Agent, Team],
        prompt: str,
        duration_seconds: Optional[float] = None,
    ) -> str:
        return self._tool.generate_sound_effect(agent, prompt, duration_seconds)

    @wrap_tool("agno_elevenlabs_text_to_speech", AgnoElevenLabsTools.text_to_speech)
    def text_to_speech(self, agent: Union[Agent, Team], prompt: str) -> str:
        return self._tool.text_to_speech(agent, prompt)
