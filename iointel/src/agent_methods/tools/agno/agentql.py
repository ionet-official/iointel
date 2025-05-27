from functools import wraps
from typing import Optional
from agno.tools.agentql import AgentQLTools as AgnoAgentQLTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class AgentQL(DisableAgnoRegistryMixin, AgnoAgentQLTools):
    def __init__(self, api_key: Optional[str] = None, agentql_query: str = ""):
        super().__init__(api_key=api_key, scrape=True, agentql_query=agentql_query)

    @register_tool(name="agentql_scrape_website")
    @wraps(AgnoAgentQLTools.scrape_website)
    def scrape_website(self, url: str) -> str:
        return super().scrape_website(url)

    @register_tool(name="agentql_custom_scrape_website")
    @wraps(AgnoAgentQLTools.custom_scrape_website)
    def custom_scrape_website(self, url: str) -> str:
        return super().custom_scrape_website(url)
