"""
Centralized tool registry utilities - single ingress point for all tool resolution
This module provides a unified interface for tool registration and resolution
"""

from typing import Union, Callable, List, Dict, Any
from ..agent_methods.data_models.datamodels import Tool
from .registries import TOOLS_REGISTRY
from ..utilities.helpers import make_logger
from .working_tools_filter import filter_available_tools, get_tool_filter_stats

logger = make_logger(__name__)

# Constants
_TYPE_MAPPING = {
    "string": "str", "integer": "int", "number": "float",
    "boolean": "bool", "array": "list", "object": "dict"
}


def _clean_type_name(type_name: str) -> str:
    """Clean and standardize type names."""
    return _TYPE_MAPPING.get(type_name, type_name)


def _extract_parameters_from_schema(json_schema: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Extract parameters from pydantic-ai generated schema."""
    properties = json_schema.get("properties", {})
    required_params = json_schema.get("required", [])
    
    parameters_with_descriptions = {}
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "any")
        param_desc = param_info.get("description", "No description available")
        default_val = param_info.get("default")
        
        param_entry = {
            "type": param_type,
            "description": param_desc,
            "required": param_name in required_params
        }
        if default_val is not None:
            param_entry["default"] = default_val
            
        parameters_with_descriptions[param_name] = param_entry
    
    return parameters_with_descriptions, required_params


def _extract_parameters_from_tool(tool: Tool) -> tuple[Dict[str, Any], List[str]]:
    """Extract parameters from tool object as fallback."""
    parameters = {}
    required_params = []
    
    if isinstance(tool.parameters, dict) and "properties" in tool.parameters:
        properties = tool.parameters["properties"]
        required_params = tool.parameters.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            if "anyOf" in param_info:
                for type_option in param_info["anyOf"]:
                    if type_option.get("type") != "null":
                        param_type = type_option.get("type", "any")
                        break
            
            parameters[param_name] = {
                "type": _clean_type_name(param_type),
                "description": param_info.get("description", "No description"),
                "required": param_name in required_params
            }
    
    return parameters, required_params


def _build_parameter_list(parameters: Dict[str, Any], required_params: List[str]) -> List[str]:
    """Build a simple parameter list for concise format."""
    param_list = []
    
    # Add required parameters first (no ?)
    for p in required_params:
        param_type = parameters.get(p, {}).get('type', 'any')
        clean_type = _clean_type_name(param_type)
        if clean_type != 'any':
            param_list.append(f"{p}: {clean_type}")
        else:
            param_list.append(p)
    
    # Add optional parameters with ? suffix
    for p in parameters:
        if p not in required_params:
            param_type = parameters.get(p, {}).get('type', 'any')
            clean_type = _clean_type_name(param_type)
            if clean_type != 'any':
                param_list.append(f"{p}: {clean_type}?")
            else:
                param_list.append(f"{p}?")
    
    return param_list


def _format_catalog_entry(tool: Tool, parameters: Dict[str, Any], required_params: List[str], 
                         verbose_format: bool, func_schema=None) -> Dict[str, Any]:
    """Format a catalog entry based on format preference."""
    if verbose_format:
        return {
            "name": tool.name,
            "description": func_schema.description if func_schema else tool.description,
            "parameters": parameters,
            "required_parameters": required_params,
            "is_async": func_schema.is_async if func_schema else tool.is_async,
            "json_schema": func_schema.json_schema if func_schema else None
        }
    else:
        brief_desc = (func_schema.description if func_schema else tool.description) or "SHOULD NOT HAPPEN"
        param_list = _build_parameter_list(parameters, required_params)
        
        return {
            "name": tool.name,
            "description": brief_desc,
            "params": param_list,
            "required": required_params
        }


def create_tool_catalog(filter_broken: bool = True, verbose_format: bool = True, debug: bool = False, use_working_filter: bool = False) -> Dict[str, Any]:
    """Create a tool catalog from available tools using pydantic-ai's schema generation. 
    This is SINGLE POINT OF TRUTH for tool descriptions and parameters, should be used for all tool resolution and documentation.
    
    Args:
        filter_broken: If True, exclude tools that fail instantiation checks
        verbose_format: If True, include full descriptions and schemas. If False, use concise format  
        debug: If True, require parameter descriptions (causes warnings). If False, skip parameter descriptions
        use_working_filter: If True, only include tools from the curated working tools list
    
    You must import tools into your session for them to register, for this to work. Or load_tools_from_env('creds') before calling this function."""
    from pydantic_ai._function_schema import function_schema
    from pydantic_ai.tools import GenerateToolJsonSchema
    
    catalog = {}
    
    # Apply working tools filter if requested
    tools_to_process = TOOLS_REGISTRY.items()
    if use_working_filter:
        available_tools = list(TOOLS_REGISTRY.keys())
        working_tools = filter_available_tools(available_tools)
        tools_to_process = [(name, tool) for name, tool in TOOLS_REGISTRY.items() if name in working_tools]
        
        # Log filtering stats
        stats = get_tool_filter_stats(available_tools)
        logger.info(f"Tool filtering: {stats['filtered_tools']}/{stats['total_available']} tools after working filter")
    
    for tool_name, tool in tools_to_process:
        # Test if tool can be instantiated
        if filter_broken:
            try:
                tool.get_wrapped_fn()
            except Exception as e:
                logger.debug(f"Skipping broken tool '{tool_name}': {e}")
                continue
        
        try:
            # Use pydantic-ai's sophisticated function schema generation
            func_schema = function_schema(
                tool.get_wrapped_fn(),
                schema_generator=GenerateToolJsonSchema,
                takes_ctx=False,
                docstring_format='auto',
                require_parameter_descriptions=debug
            )
            
            parameters, required_params = _extract_parameters_from_schema(func_schema.json_schema)
            catalog[tool_name] = _format_catalog_entry(tool, parameters, required_params, verbose_format, func_schema)
            
        except Exception:
            # Fallback to original method if pydantic-ai schema generation fails
            parameters, required_params = _extract_parameters_from_tool(tool)
            catalog[tool_name] = _format_catalog_entry(tool, parameters, required_params, verbose_format)
    
    return catalog


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
        "description": tool.description,
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
        
        debug_info["resolution_steps"].append("✅ Found in registry")
        
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
            "description": resolved_tool.description
        }
        
    except Exception as e:
        debug_info["resolution_steps"].append(f"❌ Exception: {e}")
    
    return debug_info


def create_data_source_catalog() -> Dict[str, Any]:
    """
    Create a data source catalog that matches the format expected by validation.
    This provides the schema for data_source nodes, separate from tools.
    """
    from ..agent_methods.data_models.data_source_registry import get_valid_data_source_names, get_data_source_description
    
    data_sources = {}
    
    for source_name in get_valid_data_source_names():
        if source_name == "user_input":
            data_sources[source_name] = {
                "name": source_name,
                "description": get_data_source_description(source_name),
                "parameters": {
                    "message": {
                        "type": "string",
                        "description": "The message/user query to show pass to the next step -- use this as a place to generate a sample query given the workflow",
                        "required": True
                    },
                    "default_value": {
                        "type": "string", 
                        "description": "The default value to use if the user doesn't provide one",
                        "required": True
                    }
                },
                "required_parameters": ["message", "default_value"]  # Both message and default_value are required
            }
        elif source_name == "prompt_tool":
            data_sources[source_name] = {
                "name": source_name,
                "description": get_data_source_description(source_name),
                "parameters": {
                    "message": {
                        "type": "string",
                        "description": "The message/prompt to pass to the next step",
                        "required": True
                    },
                    "default_value": {
                        "type": "string",
                        "description": "The default value to use if no message is provided",
                        "required": True
                    }
                },
                "required_parameters": ["message", "default_value"]
            }
    
    return data_sources


def create_catalog(include_tools: bool = True, include_data_sources: bool = True, **tool_kwargs) -> Dict[str, Any]:
    """
    Create a unified catalog containing tools and/or data sources.
    This is the single source of truth for workflow validation.
    
    Args:
        include_tools: Whether to include tools in the catalog
        include_data_sources: Whether to include data sources in the catalog
        **tool_kwargs: Additional arguments passed to create_tool_catalog
    
    Returns:
        Unified catalog for validation with tools and/or data sources
    """
    catalog = {}
    
    # Add tools if requested
    if include_tools:
        tool_catalog = create_tool_catalog(**tool_kwargs)
        catalog.update(tool_catalog)
    
    # Add data sources if requested
    if include_data_sources:
        data_source_catalog = create_data_source_catalog()
        catalog.update(data_source_catalog)
    
    return catalog


# Backward compatibility aliases
create_validation_catalog = create_catalog
create_unified_catalog = create_catalog


@staticmethod
def data_source_catalog_to_llm_prompt(data_source_catalog: Dict[str, Any]) -> str:
    """
    Convert data source catalog to LLM-friendly prompt format.
    Matches the pattern of tool_catalog_to_llm_prompt.
    """
    if not data_source_catalog:
        return "📋 AVAILABLE DATA SOURCES: None available"
    
    lines = [f"📋 AVAILABLE DATA SOURCES (for data_source nodes) ({len(data_source_catalog)} total):", ""]
    
    for source_name, source_info in data_source_catalog.items():
        description = source_info.get("description", "")
        required_params = source_info.get("required_parameters", [])
        all_params = source_info.get("parameters", {})
        
        lines.append(f"📦 {source_name}")
        lines.append(f"   Description: {description}")
        
        if all_params:
            lines.append("   Parameters:")
            for param_name, param_info in all_params.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required_params
                required_text = " (required)" if is_required else ""
                lines.append(f"     • {param_name} ({param_type}){required_text}: {param_desc}")
        
        lines.append(f'   Usage: "{source_name}"')
        lines.append("===")
        lines.append("")
    
    # Add dynamic templates based on actual schema
    lines.extend(["", "🚨🚨🚨 CRITICAL: DATA SOURCE CONFIG IS MANDATORY 🚨🚨🚨"])
    lines.append("Every data_source node MUST have config with ALL required parameters.")
    lines.append("")
    lines.append("COPY THESE EXACT TEMPLATES (filled in with appropriate values for the users query):")
    lines.append("")
    
    for source_name, source_info in data_source_catalog.items():
        required_params = source_info.get("required_parameters", [])
        
        # Build config example with only required parameters
        config_example = {}
        for param in required_params:
            if "prompt" in param.lower():
                config_example[param] = "Your specific prompt text here"
            elif "message" in param.lower():
                config_example[param] = "Your specific message text here"
            else:
                config_example[param] = f"Your {param} value here"
        
        # Create template
        template = {
            "type": "data_source",
            "label": "Your Label Here",
            "data": {
                "source_name": source_name,
                "config": config_example
            }
        }
        
        import json
        template_str = json.dumps(template, indent=2)
        lines.append(f"{source_name} template:")
        lines.append(template_str)
        lines.append("")
    
    required_params_list = []
    for source_name, source_info in data_source_catalog.items():
        required = source_info.get("required_parameters", [])
        if required:
            required_params_list.append(f"- {source_name}: requires {required}")
    
    if required_params_list:
        lines.extend([
            "REQUIRED PARAMETERS BY DATA SOURCE:",
            *required_params_list,
            ""
        ])
    
    lines.append("🚨 NEVER use config: null or config: {} - ALWAYS include required parameters")
    lines.append("")
    lines.append("🚨 Use these as source_name in data_source nodes. Any other names will cause failure.")
    
    return "\n".join(lines)