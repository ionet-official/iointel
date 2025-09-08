"""
Tool Grouping and Organization Utilities
========================================

Groups tools by their logical categories for better LLM comprehension.
This helps the LLM understand related tools and pick appropriate ones.
"""

from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re


def group_tools_by_category(tool_catalog: Dict[str, Any]) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Group tools by their logical category based on naming patterns.
    
    Returns a dictionary where keys are category names and values are lists of (tool_name, tool_info) tuples.
    """
    groups = defaultdict(list)
    
    for tool_name, tool_info in tool_catalog.items():
        category = _determine_category(tool_name, tool_info)
        groups[category].append((tool_name, tool_info))
    
    return dict(groups)


def _determine_category(tool_name: str, tool_info: Dict[str, Any]) -> str:
    """Determine the category for a tool based on its name and description."""
    tool_lower = tool_name.lower()
    desc_lower = tool_info.get('description', '').lower()
    
    # YFinance/Stock tools
    if any(x in tool_lower for x in ['stock', 'company', 'analyst', 'financial']):
        return "📈 Stock Market & Finance (YFinance)"
    
    # Cryptocurrency tools
    if any(x in tool_lower for x in ['coin', 'crypto', 'listing']):
        return "🪙 Cryptocurrency (CoinMarketCap)"
    
    # File operations
    if any(x in tool_lower for x in ['file_', 'csv_']):
        return "📁 File Operations"
    
    # Web/Internet tools
    if any(x in tool_lower for x in ['crawl', 'scrape', 'search_the_web', 'searxng']):
        return "🌐 Web & Search"
    
    # Academic/Research
    if 'arxiv' in tool_lower:
        return "📚 Academic Research (arXiv)"
    
    # Shell/System
    if 'shell' in tool_lower or 'command' in tool_lower:
        return "💻 System & Shell"
    
    # Math operations
    if any(x in tool_lower for x in ['add', 'subtract', 'multiply', 'divide', 'power', 'square_root', 'random']):
        return "🔢 Mathematical Operations"
    
    # Context/Memory tools
    if 'context_tree' in tool_lower:
        return "🧠 Context & Memory Management"
    
    # Control flow tools
    if 'conditional' in tool_lower or 'gate' in tool_lower:
        return "🚦 Control Flow & Routing"
    
    # Time/Weather utilities
    if 'time' in tool_lower:
        return "⏰ Time Utilities"
    if 'weather' in tool_lower:
        return "🌤️ Weather"
    
    # Retrieval/RAG tools
    if 'retrieval' in tool_lower or 'rag' in tool_lower:
        return "🔍 Retrieval & RAG"
    
    # Default category
    return "🔧 General Tools"


def format_grouped_tool_catalog(tool_catalog: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format tool catalog grouped by category for better LLM understanding.
    
    Args:
        tool_catalog: The tool catalog dictionary
        verbose: If True, include detailed parameter info
        
    Returns:
        Formatted string with tools grouped by category
    """
    grouped = group_tools_by_category(tool_catalog)
    
    sections = [
        f"# 🛠️ AVAILABLE TOOLS ({len(tool_catalog)} total)",
        "Tools are organized by category to help you find what you need:",
        ""
    ]
    
    # Sort categories for consistent presentation
    for category in sorted(grouped.keys()):
        tools = grouped[category]
        sections.append(f"\n## {category}")
        sections.append(f"{len(tools)} tool(s) available:")
        sections.append("")
        
        for tool_name, tool_info in sorted(tools, key=lambda x: x[0]):
            if verbose:
                sections.append(f"### `{tool_name}`")
                sections.append(f"**Description:** {tool_info.get('description', 'No description')}")
                
                params = tool_info.get('params', [])
                if params:
                    sections.append(f"**Parameters:** {', '.join(params)}")
                
                returns = tool_info.get('returns', '')
                if returns:
                    sections.append(f"**Returns:** {returns}")
                sections.append("")
            else:
                # Concise format
                params = tool_info.get('params', [])
                param_str = f"({', '.join(params)})" if params else "()"
                desc = tool_info.get('description', 'No description')
                
                # Truncate long descriptions
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                
                sections.append(f"  • `{tool_name}{param_str}` - {desc}")
        
        sections.append("")
    
    # Add usage notes
    sections.extend([
        "---",
        "📝 **USAGE NOTES:**",
        "• Use EXACT tool names as shown above (case-sensitive)",
        "• Tools are grouped by functionality - if you need stock data, look in the Stock Market section",
        "• If a tool is not listed here, it DOES NOT EXIST - do not hallucinate tool names",
        "• For complex workflows, combine tools from different categories",
        ""
    ])
    
    return "\n".join(sections)


def find_similar_tools(requested_tool: str, tool_catalog: Dict[str, Any], threshold: float = 0.3) -> List[str]:
    """
    Find tools similar to a requested (possibly hallucinated) tool name.
    
    Args:
        requested_tool: The tool name that was requested
        tool_catalog: Available tools
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of similar tool names that actually exist
    """
    requested_lower = requested_tool.lower()
    similar = []
    
    for actual_tool in tool_catalog.keys():
        actual_lower = actual_tool.lower()
        
        # Check for substring matches
        if requested_lower in actual_lower or actual_lower in requested_lower:
            similar.append(actual_tool)
            continue
        
        # Check for keyword matches
        requested_keywords = set(re.findall(r'\w+', requested_lower))
        actual_keywords = set(re.findall(r'\w+', actual_lower))
        
        if requested_keywords & actual_keywords:  # Intersection
            similar.append(actual_tool)
    
    return similar[:5]  # Return top 5 matches