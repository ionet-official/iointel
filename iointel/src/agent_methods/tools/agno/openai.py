from typing import Literal, Optional
from agno.tools.openai import OpenAITools as AgnoOpenAITools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent
from agno.tools.openai import OpenAIVoice, OpenAITTSModel, OpenAITTSFormat
from pydantic import Field


class OpenAI(make_base(AgnoOpenAITools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    transcription_model: str = Field(default="whisper-1", frozen=True)
    text_to_speech_voice: OpenAIVoice = Field(default="alloy", frozen=True)
    text_to_speech_model: OpenAITTSModel = Field(default="tts-1", frozen=True)
    text_to_speech_format: OpenAITTSFormat = Field(default="mp3", frozen=True)
    image_model: Optional[str] = Field(default="dall-e-3", frozen=True)
    image_quality: Optional[str] = Field(default=None, frozen=True)
    image_size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = Field(default=None, frozen=True)
    image_style: Optional[Literal["vivid", "natural"]] = Field(
        default=None, frozen=True
    )

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            enable_transcription=True,
            enable_image_generation=True,
            enable_speech_generation=True,
            transcription_model=self.transcription_model,
            text_to_speech_voice=self.text_to_speech_voice,
            text_to_speech_model=self.text_to_speech_model,
            text_to_speech_format=self.text_to_speech_format,
            image_model=self.image_model,
            image_quality=self.image_quality,
            image_size=self.image_size,
            image_style=self.image_style,
        )

    @wrap_tool("agno__openai__transcribe_audio", AgnoOpenAITools.transcribe_audio)
    def transcribe_audio(self, audio_path: str) -> str:
        return self._tool.transcribe_audio(audio_path)

    @wrap_tool("agno__openai__generate_image", AgnoOpenAITools.generate_image)
    def generate_image(self, agent: Agent, prompt: str) -> str:
        return self._tool.generate_image(agent, prompt)

    @wrap_tool("agno__openai__generate_speech", AgnoOpenAITools.generate_speech)
    def generate_speech(self, agent: Agent, text_input: str) -> str:
        return self._tool.generate_speech(agent, text_input)
