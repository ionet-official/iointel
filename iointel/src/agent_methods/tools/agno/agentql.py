from agno.tools.agentql import AgentQLTools as AgnoAgentQLTools

from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool


class AgentQL(make_base(AgnoAgentQLTools)):
    api_key: str | None = None
    agentql_query: str = ""

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key, scrape=True, agentql_query=self.agentql_query
        )

    @wrap_tool("agentql_scrape_website", AgnoAgentQLTools.scrape_website)
    def scrape_website(self, url: str) -> str:
        return self._tool.scrape_website(url)

    @wrap_tool("agentql_custom_scrape_website", AgnoAgentQLTools.custom_scrape_website)
    def custom_scrape_website(self, url: str) -> str:
        return self._tool.custom_scrape_website(url)
