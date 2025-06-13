from typing import Optional
from agno.tools.youtube import YouTubeTools as AgnoYouTubeTools
from .common import make_base, wrap_tool


class YouTube(make_base(AgnoYouTubeTools)):
    def _get_tool(self):
        return self.Inner(
            get_video_captions=self.get_video_captions_,
            get_video_data=self.get_video_data_,
            get_video_timestamps=self.get_video_timestamps,
            languages=self.languages_,
            proxies=self.proxies_,
        )

    @wrap_tool(
        "agno__youtube__get_youtube_video_id", AgnoYouTubeTools.get_youtube_video_id
    )
    def get_youtube_video_id(self, url: str) -> Optional[str]:
        return self.get_youtube_video_id(self, url)

    @wrap_tool(
        "agno__youtube__get_youtube_video_data", AgnoYouTubeTools.get_youtube_video_data
    )
    def get_youtube_video_data(self, url: str) -> str:
        return self.get_youtube_video_data(self, url)

    @wrap_tool(
        "agno__youtube__get_youtube_video_captions",
        AgnoYouTubeTools.get_youtube_video_captions,
    )
    def get_youtube_video_captions(self, url: str) -> str:
        return self.get_youtube_video_captions(self, url)

    @wrap_tool(
        "agno__youtube__get_video_timestamps", AgnoYouTubeTools.get_video_timestamps
    )
    def get_video_timestamps(self, url: str) -> str:
        return self.get_video_timestamps(self, url)
