"""
IOIntel Tools Module

This module provides a unified interface for all available tools with lazy loading
and convenient imports. Tools are organized into logical groups for easy discovery.
"""

import importlib
from typing import Any, Dict

from iointel.src.agent_methods.tools.coinmarketcap import (
    listing_coins,
    get_coin_info,
    get_coin_quotes,
    get_coin_quotes_historical,
)
from iointel.src.agent_methods.tools.context_tree import tree
from iointel.src.agent_methods.tools.duckduckgo import search_the_web
from iointel.src.agent_methods.tools.firecrawl import Crawler
from iointel.src.agent_methods.tools.retrieval_engine import RetrievalEngine
from iointel.src.agent_methods.tools.searxng import SearxngClient
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env, get_registered_tools, get_tool_by_name
from iointel.src.agent_methods.tools.utils import what_time_is_it
from iointel.src.agent_methods.tools.wolfram import Wolfram


def _lazy_import(module_path: str, name: str):
    """Lazy import helper to avoid loading all tools at startup."""
    def _import():
        module = importlib.import_module(module_path)
        return getattr(module, name)
    return _import

# Tool Clusters for convenience
class ToolClusters:
    """Convenience groupings of related tools."""
    
    @property
    def crypto(self) -> Dict[str, Any]:
        """Cryptocurrency and blockchain related tools."""
        return {
            'listing_coins': listing_coins,
            'get_coin_info': get_coin_info,
            'get_coin_quotes': get_coin_quotes,
            'get_coin_quotes_historical': get_coin_quotes_historical,
        }
    
    @property
    def search(self) -> Dict[str, Any]:
        """Search and information retrieval tools."""
        tools = {'search_the_web': search_the_web}
        for name in ['searxng_search', 'arxiv_search', 'query']:
            tool = get_tool_by_name(name)
            if tool:
                tools[name] = tool
        return tools
    
    @property
    def web(self) -> Dict[str, Any]:
        """Web scraping and crawling tools."""
        tools = {'Crawler': Crawler}
        for name in ['crawl4ai_crawl', 'agentql_query']:
            tool = get_tool_by_name(name)
            if tool:
                tools[name] = tool
        return tools
    
    @property
    def utilities(self) -> Dict[str, Any]:
        """General utility tools."""
        tools = {'what_time_is_it': what_time_is_it, 'tree': tree}
        for name in ['calculator_add', 'calculator_subtract', 'calculator_multiply', 'calculator_divide', 'yfinance_get_stock_price', 'yfinance_get_stock_info']:
            tool = get_tool_by_name(name)
            if tool:
                tools[name] = tool
        return tools

    def all_tools(self) -> Dict[str, Any]:
        """All tools."""
        return {
            **self.crypto,
            **self.search,
            **self.web,
            **self.utilities,
        }

# Create singleton instance
clusters = ToolClusters()

# Export all tools and clusters
__all__ = [
    # Core Tools
    'listing_coins',
    'get_coin_info', 
    'get_coin_quotes',
    'get_coin_quotes_historical',
    'tree',
    'search_the_web',
    'Crawler',
    'RetrievalEngine',
    'SearxngClient',
    'search_solscan_address',
    'Wolfram',
    'what_time_is_it',
    
    # Agno Tools
    'AgentQL',
    'AirflowToolkit',
    'ArxivToolkit',
    'AWSLambdaToolkit',
    'Calculator',
    'CartesiaTTS',
    'ClickUpToolkit',
    'ConfluenceToolkit',
    'Crawl4AIToolkit',
    'CSVToolkit',
    'DalleToolkit',
    'FileToolkit',
    'GoogleCalendarToolkit',
    'ShellTool',
    'YFinance',
    
    # Clusters
    'clusters',
    
    # Tool loader functions
    'load_tools_from_env',
    'get_registered_tools',
    'get_tool_by_name',
]