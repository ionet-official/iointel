"""
Data Source Registry for Valid source_name Values
================================================

This module maintains the registry of valid data source names that can be used
in data_source nodes. It provides dynamic Literal types and validation.

Only these sources provide INPUT data to workflows - everything else should be
an agent node with tools.
"""

from typing import Literal
from enum import Enum

# Registry of valid data sources
class DataSourceType(str, Enum):
    """Valid data source types for data_source nodes."""
    USER_INPUT = "user_input"          # Interactive user input collection
    PROMPT_TOOL = "prompt_tool"        # Dynamic prompt injection
    # Add new data sources here as they're created
    # WEBSOCKET_INPUT = "websocket_input"  # Future: WebSocket data streams
    # FILE_UPLOAD = "file_upload"          # Future: File upload inputs
    # SCHEDULED_TRIGGER = "scheduled_trigger"  # Future: Scheduled triggers


# Create dynamic Literal type from enum
ValidDataSourceName = Literal[
    "user_input",
    "prompt_tool"
    # This will grow as new data sources are added
]

def get_valid_data_source_names() -> list[str]:
    """Get list of all valid data source names."""
    return [source.value for source in DataSourceType]

def is_valid_data_source_name(name: str) -> bool:
    """Check if a source name is valid."""
    return name in get_valid_data_source_names()

def get_data_source_description(name: str) -> str:
    """Get description of what a data source does."""
    descriptions = {
        "user_input": "Collects interactive input from users during workflow execution. Should be the start of the workflow. Add it even if it's not specified, it will delight the user.",
        "prompt_tool": "Injects dynamic prompts or context into the workflow. Can preload the workflow with context, that the user can change. Different from user_input in that it's not the start of the workflow, and it's not required."
    }
    return descriptions.get(name, f"Unknown data source: {name}")

def create_data_source_knowledge_section() -> str:
    """Create knowledge section for WorkflowPlanner about valid data sources."""
    lines = [
        "📋 **VALID DATA SOURCES** (for data_source nodes only):",
        ""
    ]
    
    for source_type in DataSourceType:
        name = source_type.value
        description = get_data_source_description(name)
        lines.append(f"✅ **{name}**: {description}")
    
    lines.extend([
        "",
        "🚨 **CRITICAL**: Only use these exact names for data_source nodes!",
        "🚨 **API tools, web crawling, etc. = AGENT nodes with tools**",
        "🚨 **data_source = INPUT ONLY, agent = PROCESSING + TOOLS**",
        ""
    ])
    
    return "\n".join(lines)

# Example validation errors for common mistakes
COMMON_MISTAKES = {
    "crawl_the_web": "❌ 'crawl_the_web' is NOT a data source! Use: agent with tools=['Crawler-scrape_url']",
    "web_search": "❌ 'web_search' is NOT a data source! Use: agent with tools=['searxng.search']",
    "stock_api": "❌ 'stock_api' is NOT a data source! Use: agent with tools=['get_current_stock_price']",
    "weather_api": "❌ 'weather_api' is NOT a data source! Use: agent with tools=['get_weather']",
    "email_tool": "❌ 'email_tool' is NOT a data source! Use: agent with tools=['send_email']",
}

def get_correction_for_mistake(invalid_name: str) -> str:
    """Get correction suggestion for common mistakes."""
    return COMMON_MISTAKES.get(invalid_name, 
        f"❌ '{invalid_name}' is NOT a valid data source! "
        f"Valid sources: {get_valid_data_source_names()}. "
        f"For tools/APIs, use agent nodes instead."
    )