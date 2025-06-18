from typing import Any, Dict, List, Optional
from agno.tools.youtube import YouTubeTools as AgnoYouTubeTools
from .common import make_base, wrap_tool
from pydantic import Field


class YouTube(make_base(AgnoYouTubeTools)):
    languages: Optional[List[str]] = Field(default=None, frozen=True)
    proxies: Optional[Dict[str, Any]] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            get_video_captions=True,
            get_video_data=True,
            get_video_timestamps=True,
            languages=self.languages,
            proxies=self.proxies,
        )

    @wrap_tool(
        "agno__youtube__get_youtube_video_id", AgnoYouTubeTools.get_youtube_video_id
    )
    def get_youtube_video_id(self, url: str) -> Optional[str]:
        return self._tool.get_youtube_video_id(url)

    @wrap_tool(
        "agno__youtube__get_youtube_video_data", AgnoYouTubeTools.get_youtube_video_data
    )
    def get_youtube_video_data(self, url: str) -> str:
        return self._tool.get_youtube_video_data(url)

    @wrap_tool(
        "agno__youtube__get_youtube_video_captions",
        AgnoYouTubeTools.get_youtube_video_captions,
    )
    def get_youtube_video_captions(self, url: str) -> str:
        return self._tool.get_youtube_video_captions(url)

    @wrap_tool(
        "agno__youtube__get_video_timestamps", AgnoYouTubeTools.get_video_timestamps
    )
    def get_video_timestamps(self, url: str) -> str:
        return self._tool.get_video_timestamps(url)
