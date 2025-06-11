from agno.tools.mcp import MCPTools as AgnoMCPTools
from .common import make_base, wrap_tool


class MCP(make_base(AgnoMCPTools)):
    def _get_tool(self):
        return self.Inner(
            command=self.command_,
            url=self.url_,
            env=self.env_,
            transport=self.transport_,
            server_params=self.server_params_,
            session=self.session_,
            timeout_seconds=self.timeout_seconds_,
            client=self.client_,
            include_tools=self.include_tools_,
            exclude_tools=self.exclude_tools_,
        )

    @wrap_tool("agno__mcp__initialize", AgnoMCPTools.initialize)
    def initialize(self) -> None:
        return self.initialize(self)
