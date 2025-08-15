"""
Working Tools Filter for WorkflowPlanner

This module provides a curated list of tools that are known to work properly
with the WorkflowPlanner, filtering out problematic tools that cause pydantic_ai
issues or other compatibility problems.

Based on testing with Llama models, these tools have been verified to work.
"""

from typing import List, Set

# Tools that are confirmed working from Llama testing
WORKING_TOOLS = {
    # Crypto/Financial
    "listing_coins",
    "get_coin_info", 
    "get_coin_quotes",
    "get_coin_quotes_historical",
    
    # Core utilities
    "what_time_is_it",
    "user_input",
    "prompt_tool",
    "conditional_gate",
    "conditional_multi_gate",
    
    # Basic math tools (cleaner than agno calculator tools)
    "add",
    "subtract", 
    "multiply",
    "divide",
    "square_root",
    
    # Calculator tools (agno - keep for backward compatibility)
    "calculator_add",
    "calculator_subtract", 
    "calculator_multiply",
    "calculator_divide",
    "calculator_exponentiate",
    "calculator_square_root",
    "calculator_factorial",
    "calculator_is_prime",
    
    # File operations
    "airflow_save_dag_file",
    "run_shell_command",
    "file_list",
    "file_save",
    
    # Stock/Finance tools
    "get_analyst_recommendations",
    "get_company_info",
    "get_company_news", 
    "get_current_stock_price",
    "get_historical_stock_prices",
    "get_income_statements",
    "get_key_financial_ratios",
    "get_stock_fundamentals",
    "get_technical_indicators",
    
    # Web crawling
    "agno__crawl4ai__web_crawler",
    
    # CSV tools (work if files exist)
    "csv_get_columns",
    "csv_list_csv_files", 
    "csv_query_csv_file",
    "csv_read_csv_file",
    
    # File reading (works if files exist)
    "file_read",
    "airflow_read_dag_file",
}

# Tools that cause pydantic_ai KeyError: 'self' issues
PROBLEMATIC_TOOLS = {
    "create_image",      # dalle.py - uses Agent/Team types
    "add_memory",        # mem0.py - uses Agent types
    "delete_all_memories", # mem0.py - uses Agent types  
    "get_all_memories",  # mem0.py - uses Agent types
    "search_memory",     # mem0.py - uses Agent types
}

# Tools that don't work reliably (missing APIs, credentials, etc.)
UNRELIABLE_TOOLS = {
    "search_the_web",        # API issues
    "search_the_web_async",  # API issues
    "arxiv_read_papers",     # Sometimes fails
    "arxiv_search",          # Sometimes fails
}


def get_working_tools_for_planner() -> List[str]:
    """
    Get a curated list of tools that work reliably with WorkflowPlanner.
    
    Returns:
        List of tool names that are safe to use in workflows
    """
    return sorted(list(WORKING_TOOLS))


def filter_available_tools(available_tools: List[str]) -> List[str]:
    """
    Filter available tools to only include those that work reliably.
    
    Args:
        available_tools: List of all available tool names from discovery
        
    Returns:
        Filtered list containing only working tools
    """
    available_set = set(available_tools)
    working_set = WORKING_TOOLS & available_set  # Intersection
    
    return sorted(list(working_set))


def is_working_tool(tool_name: str) -> bool:
    """
    Check if a tool is in the working tools list.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        True if the tool is known to work reliably
    """
    return tool_name in WORKING_TOOLS


def get_tool_filter_stats(available_tools: List[str]) -> dict:
    """
    Get statistics about tool filtering.
    
    Args:
        available_tools: List of all available tool names
        
    Returns:
        Dictionary with filtering statistics
    """
    available_set = set(available_tools)
    
    working_count = len(WORKING_TOOLS & available_set)
    problematic_count = len(PROBLEMATIC_TOOLS & available_set)
    unreliable_count = len(UNRELIABLE_TOOLS & available_set)
    other_count = len(available_set - WORKING_TOOLS - PROBLEMATIC_TOOLS - UNRELIABLE_TOOLS)
    
    return {
        "total_available": len(available_tools),
        "working_tools": working_count,
        "problematic_tools": problematic_count,
        "unreliable_tools": unreliable_count,
        "other_tools": other_count,
        "filtered_tools": working_count
    }


# Quick access function for WorkflowPlanner
def get_curated_tool_list() -> List[str]:
    """
    Get the curated list of working tools for WorkflowPlanner use.
    This is the main function to use when you need reliable tools.
    
    Returns:
        List of tool names that work reliably
    """
    return get_working_tools_for_planner()