from typing import Optional, List, Dict
from agno.tools.moviepy_video import MoviePyVideoTools as AgnoMoviePyVideoTools
from .common import make_base, wrap_tool
from moviepy.video.VideoClip import TextClip
from pydantic import Field


class MoviePyVideo(make_base(AgnoMoviePyVideoTools)):
    process_video: bool = Field(default=True, frozen=True)
    generate_captions: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            process_video=self.process_video,
            generate_captions=self.generate_captions,
            embed_captions=True,
        )

    @wrap_tool(
        "agno__moviepyvideo__split_text_into_lines",
        AgnoMoviePyVideoTools.split_text_into_lines,
    )
    def split_text_into_lines(self, words: List[Dict]) -> List[Dict]:
        return self._tool.split_text_into_lines(words)

    @wrap_tool(
        "agno__moviepyvideo__create_caption_clips",
        AgnoMoviePyVideoTools.create_caption_clips,
    )
    def create_caption_clips(
        self,
        text_json: Dict,
        frame_size: tuple,
        font="Arial",
        color="white",
        highlight_color="yellow",
        stroke_color="black",
        stroke_width=1.5,
    ) -> List[TextClip]:
        return self._tool.create_caption_clips(
            text_json,
            frame_size,
            font,
            color,
            highlight_color,
            stroke_color,
            stroke_width,
        )

    @wrap_tool("agno__moviepyvideo__parse_srt", AgnoMoviePyVideoTools.parse_srt)
    def parse_srt(self, srt_content: str) -> List[Dict]:
        return self._tool.parse_srt(srt_content)

    @wrap_tool("agno__moviepyvideo__extract_audio", AgnoMoviePyVideoTools.extract_audio)
    def extract_audio(self, video_path: str, output_path: str) -> str:
        return self._tool.extract_audio(video_path, output_path)

    @wrap_tool("agno__moviepyvideo__create_srt", AgnoMoviePyVideoTools.create_srt)
    def create_srt(self, transcription: str, output_path: str) -> str:
        return self._tool.create_srt(transcription, output_path)

    @wrap_tool(
        "agno__moviepyvideo__embed_captions", AgnoMoviePyVideoTools.embed_captions
    )
    def embed_captions(
        self,
        video_path: str,
        srt_path: str,
        output_path: Optional[str] = None,
        font_size: int = 24,
        font_color: str = "white",
        stroke_color: str = "black",
        stroke_width: int = 1,
    ) -> str:
        return self._tool.embed_captions(
            video_path,
            srt_path,
            output_path,
            font_size,
            font_color,
            stroke_color,
            stroke_width,
        )
