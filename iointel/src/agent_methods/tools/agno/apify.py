from typing import List, Optional, Union
from agno.tools.apify import ApifyTools as AgnoApifyTools
from pydantic import Field
from .common import make_base, wrap_tool


class Apify(make_base(AgnoApifyTools)):
    actors: Optional[Union[str, List[str]]] = Field(default=None, frozen=True)
    apify_api_token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            actors=self.actors,
            apify_api_token=self.apify_api_token,
        )

    @wrap_tool("apify_register_actor", AgnoApifyTools.make_request)
    def register_actor(self, actor_id: str) -> None:
        return self._tool.register_actor(actor_id)
