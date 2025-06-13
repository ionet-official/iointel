from agno.tools.openai import OpenAITools as AgnoOpenAITools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent


class OpenAI(make_base(AgnoOpenAITools)):
    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key_,
            enable_transcription=self.enable_transcription_,
            enable_image_generation=self.enable_image_generation_,
            enable_speech_generation=self.enable_speech_generation_,
            transcription_model=self.transcription_model_,
            text_to_speech_voice=self.text_to_speech_voice_,
            text_to_speech_model=self.text_to_speech_model_,
            text_to_speech_format=self.text_to_speech_format_,
            image_model=self.image_model_,
            image_quality=self.image_quality_,
            image_size=self.image_size_,
            image_style=self.image_style_,
        )

    @wrap_tool("agno__openai__transcribe_audio", AgnoOpenAITools.transcribe_audio)
    def transcribe_audio(self, audio_path: str) -> str:
        return self.transcribe_audio(self, audio_path)

    @wrap_tool("agno__openai__generate_image", AgnoOpenAITools.generate_image)
    def generate_image(self, agent: Agent, prompt: str) -> str:
        return self._tool.generate_image(agent, prompt)

    @wrap_tool("agno__openai__generate_speech", AgnoOpenAITools.generate_speech)
    def generate_speech(self, agent: Agent, text_input: str) -> str:
        return self._tool.generate_speech(agent, text_input)
