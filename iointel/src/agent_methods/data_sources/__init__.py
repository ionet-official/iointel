"""
Data Sources Module
==================

This module contains data source implementations that provide INPUT data to workflows.
These are NOT tools - they are input providers that feed data to agent nodes.

Data sources are registered separately from tools and have a standardized interface.
"""

from iointel.src.agent_methods.data_sources.input_sources import user_input_source, prompt_source
from iointel.src.agent_methods.data_sources.registry import register_data_source, get_data_source, list_data_sources

__all__ = [
    "user_input_source",
    "prompt_source", 
    "register_data_source",
    "get_data_source",
    "list_data_sources"
]