from agno.tools.mlx_transcribe import MLXTranscribeTools as AgnoMLXTranscribeTools
from .common import make_base, wrap_tool


class MLXTranscribe(make_base(AgnoMLXTranscribeTools)):
    def _get_tool(self):
        return self.Inner(
            base_dir=self.base_dir_,
            read_files_in_base_dir=self.read_files_in_base_dir_,
            path_or_hf_repo=self.path_or_hf_repo_,
            verbose=self.verbose_,
            temperature=self.temperature_,
            compression_ratio_threshold=self.compression_ratio_threshold_,
            logprob_threshold=self.logprob_threshold_,
            no_speech_threshold=self.no_speech_threshold_,
            condition_on_previous_text=self.condition_on_previous_text_,
            initial_prompt=self.initial_prompt_,
            word_timestamps=self.word_timestamps_,
            prepend_punctuations=self.prepend_punctuations_,
            append_punctuations=self.append_punctuations_,
            clip_timestamps=self.clip_timestamps_,
            hallucination_silence_threshold=self.hallucination_silence_threshold_,
            decode_options=self.decode_options_,
        )

    @wrap_tool("agno__mlxtranscribe__transcribe", AgnoMLXTranscribeTools.transcribe)
    def transcribe(self, file_name: str) -> str:
        return self.transcribe(self, file_name)

    @wrap_tool("agno__mlxtranscribe__read_files", AgnoMLXTranscribeTools.read_files)
    def read_files(self) -> str:
        return self.read_files(self)
