from typing import Optional, Union
from agno.tools.giphy import GiphyTools as AgnoGiphyTools
from agno.agent import Agent
from agno.team import Team

from .common import make_base, wrap_tool
from pydantic import Field


class Giphy(make_base(AgnoGiphyTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    limit: int = Field(default=1, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            limit=self.limit,
        )

    @wrap_tool("agno__giphy__search_gifs", AgnoGiphyTools.search_gifs)
    def search_gifs(self, agent: Union[Agent, Team], query: str) -> str:
        return self._tool.search_gifs(agent, query)
