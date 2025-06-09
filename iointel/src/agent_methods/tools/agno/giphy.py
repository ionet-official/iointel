from typing import Union
from agno.tools.giphy import GiphyTools as AgnoGiphyTools
from agno.agent import Agent
from agno.team import Team

from .common import make_base, wrap_tool


class Giphy(make_base(AgnoGiphyTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("agno__giphy__search_gifs", AgnoGiphyTools.search_gifs)
    def search_gifs(self, agent: Union[Agent, Team], query: str) -> str:
        return self._tool.search_gifs(agent, query)
