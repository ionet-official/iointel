from pathlib import Path
from typing import List, Optional, Tuple, Union
from agno.tools.mlx_transcribe import MLXTranscribeTools as AgnoMLXTranscribeTools
from .common import make_base, wrap_tool
from pydantic import Field


class MLXTranscribe(make_base(AgnoMLXTranscribeTools)):
    base_dir: Optional[Path] = Field(default=None, frozen=True)
    path_or_hf_repo: str = ("mlx-community/whisper-large-v3-turbo",)
    verbose: Optional[bool] = Field(default=None, frozen=True)
    temperature: Optional[Union[float, Tuple[float, ...]]] = Field(
        default=None, frozen=True
    )
    compression_ratio_threshold: Optional[float] = Field(default=None, frozen=True)
    logprob_threshold: Optional[float] = Field(default=None, frozen=True)
    no_speech_threshold: Optional[float] = Field(default=None, frozen=True)
    condition_on_previous_text: Optional[bool] = Field(default=None, frozen=True)
    initial_prompt: Optional[str] = Field(default=None, frozen=True)
    word_timestamps: Optional[bool] = Field(default=None, frozen=True)
    prepend_punctuations: Optional[str] = Field(default=None, frozen=True)
    append_punctuations: Optional[str] = Field(default=None, frozen=True)
    clip_timestamps: Optional[Union[str, List[float]]] = Field(
        default=None, frozen=True
    )
    hallucination_silence_threshold: Optional[float] = Field(default=None, frozen=True)
    decode_options: Optional[dict] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            base_dir=self.base_dir,
            read_files_in_base_dir=True,
            path_or_hf_repo=self.path_or_hf_repo,
            verbose=self.verbose,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            logprob_threshold=self.logprob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            word_timestamps=self.word_timestamps,
            prepend_punctuations=self.prepend_punctuations,
            append_punctuations=self.append_punctuations,
            clip_timestamps=self.clip_timestamps,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            decode_options=self.decode_options,
        )

    @wrap_tool("agno__mlxtranscribe__transcribe", AgnoMLXTranscribeTools.transcribe)
    def transcribe(self, file_name: str) -> str:
        return self.transcribe(file_name)

    @wrap_tool("agno__mlxtranscribe__read_files", AgnoMLXTranscribeTools.read_files)
    def read_files(self) -> str:
        return self.read_files()
