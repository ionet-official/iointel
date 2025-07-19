"""
Centralized tool registry utilities - single ingress point for all tool resolution
This module provides a unified interface for tool registration and resolution
"""

from typing import Union, Callable, Optional, List
from ..agent_methods.data_models.datamodels import Tool
from .registries import TOOLS_REGISTRY
from ..utilities.helpers import make_logger

logger = make_logger(__name__)


def resolve_tool(tool_data: Union[str, Tool, Callable], allow_unregistered: bool = False) -> Tool:
    """
    Unified tool resolution function - single ingress point for all tool resolution.
    
    This function handles all the complexities of tool resolution including:
    - String tool name lookups
    - Tool object validation
    - Callable function wrapping
    - Agno tool body comparison issues
    - Registry conflicts
    
    Args:
        tool_data: Can be a string tool name, Tool object, or callable function
        allow_unregistered: Whether to allow unregistered tools
        
    Returns:
        Tool: The resolved tool object
        
    Raises:
        ValueError: If tool cannot be resolved
    """
    logger.debug(f"Resolving tool: {tool_data} (type: {type(tool_data)})")
    
    # Case 1: String tool name - lookup in registry
    if isinstance(tool_data, str):
        tool_name = tool_data
        logger.debug(f"String lookup for tool: {tool_name}")
        
        if tool_name not in TOOLS_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        registered_tool = TOOLS_REGISTRY[tool_name]
        logger.debug(f"Found tool: {registered_tool.name} -> {registered_tool.fn.__name__}")
        
        # For string lookups, return the exact tool from registry
        # This avoids the body comparison issues that cause agno tool conflicts
        return registered_tool
    
    # Case 2: Already a Tool object
    elif isinstance(tool_data, Tool):
        logger.debug(f"Tool object provided: {tool_data.name}")
        return tool_data
    
    # Case 3: Callable function - convert to Tool
    elif callable(tool_data):
        logger.debug(f"Callable provided: {tool_data}")
        return Tool.from_function(tool_data)
    
    # Case 4: Invalid type
    else:
        raise ValueError(f"Invalid tool type: {type(tool_data)}. Expected str, Tool, or callable")


def resolve_tools(tool_list: List[Union[str, Tool, Callable]], allow_unregistered: bool = False) -> List[Tool]:
    """
    Resolve a list of tools using the unified resolver.
    
    Args:
        tool_list: List of tool identifiers (strings, Tool objects, or callables)
        allow_unregistered: Whether to allow unregistered tools
        
    Returns:
        List[Tool]: List of resolved Tool objects
    """
    logger.debug(f"Resolving {len(tool_list)} tools")
    
    resolved_tools = []
    for i, tool_data in enumerate(tool_list):
        try:
            resolved_tool = resolve_tool(tool_data, allow_unregistered)
            resolved_tools.append(resolved_tool)
            logger.debug(f"Resolved tool {i+1}/{len(tool_list)}: {resolved_tool.name}")
        except Exception as e:
            logger.error(f"Failed to resolve tool {i+1}/{len(tool_list)}: {tool_data} - {e}")
            if not allow_unregistered:
                raise
    
    return resolved_tools


def get_tool_info(tool_name: str) -> dict:
    """
    Get diagnostic information about a tool.
    
    Args:
        tool_name: Name of the tool to inspect
        
    Returns:
        dict: Tool information including name, function, registry status, etc.
    """
    if tool_name not in TOOLS_REGISTRY:
        return {"exists": False, "name": tool_name}
    
    tool = TOOLS_REGISTRY[tool_name]
    return {
        "exists": True,
        "name": tool.name,
        "registry_key": tool_name,
        "function_name": tool.fn.__name__,
        "qualified_name": getattr(tool.fn, '__qualname__', 'unknown'),
        "description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
        "is_async": tool.is_async,
        "has_body": tool.body is not None,
        "body_length": len(tool.body) if tool.body else 0,
        "name_matches_function": tool.name == tool.fn.__name__,
        "registry_key_matches_name": tool_name == tool.name,
    }


def validate_tool_registry() -> dict:
    """
    Validate the tool registry for common issues.
    
    Returns:
        dict: Validation results with any issues found
    """
    issues = []
    total_tools = len(TOOLS_REGISTRY)
    
    # Check for name mismatches
    name_mismatches = []
    function_mismatches = []
    
    for registry_key, tool in TOOLS_REGISTRY.items():
        # Check if registry key matches tool name
        if registry_key != tool.name:
            name_mismatches.append({
                "registry_key": registry_key,
                "tool_name": tool.name,
                "function_name": tool.fn.__name__
            })
        
        # Check if tool name matches function name
        if tool.name != tool.fn.__name__:
            function_mismatches.append({
                "tool_name": tool.name,
                "function_name": tool.fn.__name__,
                "registry_key": registry_key
            })
    
    if name_mismatches:
        issues.append(f"Registry key mismatches: {len(name_mismatches)} tools")
    
    if function_mismatches:
        issues.append(f"Function name mismatches: {len(function_mismatches)} tools")
    
    return {
        "total_tools": total_tools,
        "issues": issues,
        "name_mismatches": name_mismatches,
        "function_mismatches": function_mismatches,
        "is_healthy": len(issues) == 0
    }


def debug_tool_resolution(tool_name: str) -> dict:
    """
    Debug tool resolution for a specific tool.
    
    Args:
        tool_name: Name of the tool to debug
        
    Returns:
        dict: Debug information about the tool resolution process
    """
    debug_info = {
        "tool_name": tool_name,
        "resolution_steps": [],
        "final_result": None,
        "success": False
    }
    
    try:
        # Step 1: Check if tool exists in registry
        debug_info["resolution_steps"].append(f"Checking registry for '{tool_name}'")
        
        if tool_name not in TOOLS_REGISTRY:
            debug_info["resolution_steps"].append(f"❌ Tool '{tool_name}' not found in registry")
            return debug_info
        
        debug_info["resolution_steps"].append(f"✅ Found in registry")
        
        # Step 2: Get tool from registry
        tool = TOOLS_REGISTRY[tool_name]
        debug_info["resolution_steps"].append(f"Registry tool: name='{tool.name}', fn='{tool.fn.__name__}'")
        
        # Step 3: Resolve using unified resolver
        resolved_tool = resolve_tool(tool_name)
        debug_info["resolution_steps"].append(f"Resolved tool: name='{resolved_tool.name}', fn='{resolved_tool.fn.__name__}'")
        
        # Step 4: Validate resolution
        if resolved_tool.name == tool_name and resolved_tool.fn.__name__ == tool_name:
            debug_info["resolution_steps"].append("✅ Resolution successful - names match")
            debug_info["success"] = True
        else:
            debug_info["resolution_steps"].append("❌ Resolution failed - name mismatch")
        
        debug_info["final_result"] = {
            "name": resolved_tool.name,
            "function": resolved_tool.fn.__name__,
            "description": resolved_tool.description[:50] + "..." if len(resolved_tool.description) > 50 else resolved_tool.description
        }
        
    except Exception as e:
        debug_info["resolution_steps"].append(f"❌ Exception: {e}")
    
    return debug_info