from typing import Literal, Optional, Union
from agno.tools.mcp import MCPTools as AgnoMCPTools
from mcp import (
    ClientSession,
    StdioServerParameters,
    SSEClientParams,
    StreamableHTTPClientParams,
)
from .common import make_base, wrap_tool
from pydantic import Field


class MCP(make_base(AgnoMCPTools)):
    command: Optional[str] = Field(default=None, frozen=True)
    url: Optional[str] = Field(default=None, frozen=True)
    env: Optional[dict[str, str]] = Field(default=None, frozen=True)
    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        default="stdio", frozen=True
    )
    server_params: Optional[
        Union[StdioServerParameters, SSEClientParams, StreamableHTTPClientParams]
    ] = Field(default=None, frozen=True)
    session: Optional[ClientSession] = Field(default=None, frozen=True)
    timeout_seconds: int = Field(default=5, frozen=True)
    client = Field(default=None, frozen=True)
    include_tools: Optional[list[str]] = Field(default=None, frozen=True)
    exclude_tools: Optional[list[str]] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            command=self.command,
            url=self.url,
            env=self.env,
            transport=self.transport,
            server_params=self.server_params,
            session=self.session,
            timeout_seconds=self.timeout_seconds,
            client=self.client,
            include_tools=self.include_tools,
            exclude_tools=self.exclude_tools,
        )

    @wrap_tool("agno__mcp__initialize", AgnoMCPTools.initialize)
    def initialize(self) -> None:
        return self.initialize(self)
