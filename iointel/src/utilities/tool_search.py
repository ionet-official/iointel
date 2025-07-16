"""
Advanced tool search functionality for iointel.

This module provides fuzzy matching, semantic search, and categorization
capabilities for discovering and selecting tools.
"""

import difflib
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import re


# Tool categories for better organization
TOOL_CATEGORIES = {
    "search": ["duckduckgo", "google_search", "tavily", "searxng", "exa"],
    "communication": ["email", "slack", "discord", "telegram", "whatsapp"],
    "data": ["csv", "pandas", "sql", "duckdb", "google_sheets"],
    "ai": ["openai", "cartesia", "elevenlabs", "dalle", "fal"],
    "file": ["file", "localfilesystem"],
    "web": ["firecrawl", "crawl4ai", "browserbase", "spider"],
    "social": ["reddit", "x", "hackernews"],
    "productivity": ["jira", "linear", "trello", "todoist", "clickup"],
    "cloud": ["aws_lambda", "google_bigquery", "google_calendar"],
    "media": ["youtube", "giphy", "moviepyvideo", "lumalab"],
    "finance": ["yfinance", "financial_datasets"],
    "research": ["arxiv", "pubmed", "wikipedia", "newspaper"],
    "utility": ["calculator", "sleep", "reasoning", "python", "shell"],
}


def fuzzy_search_tools(query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
    """
    Search for tools using fuzzy string matching.
    
    Args:
        query: Search query string
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of (tool_name, similarity_score) tuples sorted by score
    """
    from .registries import TOOLS_REGISTRY
    
    matches = []
    query_lower = query.lower()
    
    for tool_name in TOOLS_REGISTRY.keys():
        tool_name_lower = tool_name.lower()
        
        # Direct substring match gets higher score
        if query_lower in tool_name_lower:
            score = 0.9 + (len(query_lower) / len(tool_name_lower)) * 0.1
            matches.append((tool_name, min(score, 1.0)))
            continue
        
        # Fuzzy string matching
        ratio = difflib.SequenceMatcher(None, query_lower, tool_name_lower).ratio()
        if ratio >= threshold:
            matches.append((tool_name, ratio))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)


def search_tools_by_description(query: str, min_score: float = 0.3) -> List[Tuple[str, float]]:
    """
    Search tools by their docstring/description content.
    
    Args:
        query: Search query string
        min_score: Minimum relevance score
    
    Returns:
        List of (tool_name, relevance_score) tuples
    """
    from .registries import TOOLS_REGISTRY
    
    results = []
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    for tool_name, tool in TOOLS_REGISTRY.items():
        if not tool.description:
            continue
        
        description_lower = tool.description.lower()
        description_words = set(re.findall(r'\b\w+\b', description_lower))
        
        # Calculate relevance score based on word overlap
        common_words = query_words.intersection(description_words)
        if not common_words:
            continue
        
        # Score based on overlap and query coverage
        overlap_score = len(common_words) / len(query_words)
        coverage_score = len(common_words) / len(description_words)
        relevance_score = (overlap_score * 0.7) + (coverage_score * 0.3)
        
        # Boost score for exact phrase matches
        if query_lower in description_lower:
            relevance_score = min(relevance_score * 1.5, 1.0)
        
        if relevance_score >= min_score:
            results.append((tool_name, relevance_score))
    
    return sorted(results, key=lambda x: x[1], reverse=True)


def get_tools_by_category(category: str) -> List[str]:
    """
    Get all tools in a specific category.
    
    Args:
        category: Category name (e.g., "search", "communication")
    
    Returns:
        List of tool names in the category
    """
    from .registries import TOOLS_REGISTRY
    
    category_tools = TOOL_CATEGORIES.get(category.lower(), [])
    return [tool for tool in category_tools if tool in TOOLS_REGISTRY]


def get_all_categories() -> Dict[str, List[str]]:
    """
    Get all available tool categories and their tools.
    
    Returns:
        Dictionary mapping category names to lists of tool names
    """
    from .registries import TOOLS_REGISTRY
    
    result = {}
    for category, tools in TOOL_CATEGORIES.items():
        available_tools = [tool for tool in tools if tool in TOOLS_REGISTRY]
        if available_tools:
            result[category] = available_tools
    
    return result


def suggest_similar_tools(tool_name: str, limit: int = 5) -> List[str]:
    """
    Suggest similar tools based on name and category.
    
    Args:
        tool_name: Name of the tool to find similar tools for
        limit: Maximum number of suggestions
    
    Returns:
        List of similar tool names
    """
    from .registries import TOOLS_REGISTRY
    
    if tool_name not in TOOLS_REGISTRY:
        return []
    
    # Find the category of the input tool
    tool_category = None
    for category, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            tool_category = category
            break
    
    suggestions = []
    
    # Add tools from the same category
    if tool_category:
        category_tools = get_tools_by_category(tool_category)
        suggestions.extend([t for t in category_tools if t != tool_name])
    
    # Add fuzzy matches
    fuzzy_matches = fuzzy_search_tools(tool_name, threshold=0.4)
    for match_name, _ in fuzzy_matches:
        if match_name != tool_name and match_name not in suggestions:
            suggestions.append(match_name)
    
    return suggestions[:limit]


def smart_tool_search(
    query: str,
    search_descriptions: bool = True,
    search_fuzzy: bool = True,
    fuzzy_threshold: float = 0.6,
    limit: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Comprehensive smart tool search combining multiple methods.
    
    Args:
        query: Search query
        search_descriptions: Whether to search tool descriptions
        search_fuzzy: Whether to use fuzzy matching
        fuzzy_threshold: Minimum fuzzy match score
        limit: Maximum results to return
    
    Returns:
        List of (tool_name, score, match_type) tuples
    """
    from .registries import TOOLS_REGISTRY
    
    results = []
    seen_tools = set()
    
    # 1. Exact name match (highest priority)
    if query in TOOLS_REGISTRY:
        results.append((query, 1.0, "exact"))
        seen_tools.add(query)
    
    # 2. Category search
    if query.lower() in TOOL_CATEGORIES:
        category_tools = get_tools_by_category(query.lower())
        for tool in category_tools:
            if tool not in seen_tools:
                results.append((tool, 0.95, "category"))
                seen_tools.add(tool)
    
    # 3. Fuzzy name matching
    if search_fuzzy:
        fuzzy_matches = fuzzy_search_tools(query, fuzzy_threshold)
        for tool_name, score in fuzzy_matches:
            if tool_name not in seen_tools:
                results.append((tool_name, score * 0.8, "fuzzy"))
                seen_tools.add(tool_name)
    
    # 4. Description search
    if search_descriptions:
        desc_matches = search_tools_by_description(query)
        for tool_name, score in desc_matches:
            if tool_name not in seen_tools:
                results.append((tool_name, score * 0.7, "description"))
                seen_tools.add(tool_name)
    
    # Sort by score and return top results
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def get_tool_info(tool_name: str) -> Optional[Dict[str, any]]:
    """
    Get comprehensive information about a tool.
    
    Args:
        tool_name: Name of the tool
    
    Returns:
        Dictionary with tool information or None if not found
    """
    from .registries import TOOLS_REGISTRY
    
    if tool_name not in TOOLS_REGISTRY:
        return None
    
    tool = TOOLS_REGISTRY[tool_name]
    
    # Find tool category
    category = None
    for cat, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            category = cat
            break
    
    return {
        "name": tool.name,
        "description": tool.description,
        "category": category,
        "parameters": tool.parameters,
        "similar_tools": suggest_similar_tools(tool_name),
        "has_body": bool(tool.body),
        "is_stateful": bool(tool.fn_metadata and tool.fn_metadata.stateful),
    }