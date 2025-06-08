from typing import Optional, Union
from agno.tools.desi_vocal import DesiVocalTools as AgnoDesiVocalTools
from agno.agent import Agent
from agno.team.team import Team

from .common import make_base, wrap_tool


class DesiVocal(make_base(AgnoDesiVocalTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("calculator_add", AgnoDesiVocalTools.add)
    def get_voices(self) -> str:
        return self._tool.get_voices()

    def text_to_speech(
        self, agent: Union[Agent, Team], prompt: str, voice_id: Optional[str] = None
    ) -> str:
        return self._tool.get_voices()
