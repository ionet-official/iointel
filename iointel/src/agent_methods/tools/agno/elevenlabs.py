# from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools
from typing import Optional, Union
from agno.tools.eleven_labs import ElevenLabsTools as AgnoElevenLabsTools
from agno.agent import Agent
from agno.team import Team
from .common import make_base, wrap_tool


class ElevenLabs(make_base(AgnoElevenLabsTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

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
