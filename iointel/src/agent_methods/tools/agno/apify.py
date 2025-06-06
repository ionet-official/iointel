from agno.tools.apify import ApifyTools as AgnoApifyTools

from .common import make_base, wrap_tool


class Api(make_base(AgnoApifyTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("apify_register_actor", AgnoApifyTools.make_request)
    def register_actor(self, actor_id: str) -> None:
        return self._tool.register_actor(actor_id)
