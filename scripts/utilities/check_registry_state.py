#!/usr/bin/env python3
"""
Check the exact state of the registry for the problematic tools
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY

# Load tools
tools = load_tools_from_env()

# Check the exact state
print("Registry state for problematic tools:")
print("=" * 50)

for tool_name in ["run_shell_command", "get_analyst_recommendations"]:
    if tool_name in TOOLS_REGISTRY:
        tool = TOOLS_REGISTRY[tool_name]
        print(f"\n{tool_name}:")
        print(f"  Registry key: {tool_name}")
        print(f"  Tool.name: {tool.name}")
        print(f"  Tool.fn: {tool.fn}")
        print(f"  Tool.fn.__name__: {tool.fn.__name__}")
        
        # Check if the registry key matches the tool name
        if tool_name != tool.name:
            print(f"  ❌ MISMATCH: Registry key '{tool_name}' has tool with name '{tool.name}'")
        else:
            print("  ✅ MATCH: Registry key and tool name match")
            
        # Check if the function name matches the tool name
        if tool.fn.__name__ != tool.name:
            print(f"  ❌ MISMATCH: Tool name '{tool.name}' has function '{tool.fn.__name__}'")
        else:
            print("  ✅ MATCH: Tool name and function name match")
    else:
        print(f"\n{tool_name}: NOT FOUND in registry")

print(f"\nTotal tools in registry: {len(TOOLS_REGISTRY)}")

# Check if there are any tools with mismatched names
print("\nChecking for name mismatches across all tools:")
mismatches = []
for registry_key, tool in TOOLS_REGISTRY.items():
    if registry_key != tool.name:
        mismatches.append((registry_key, tool.name, tool.fn.__name__))

if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    for registry_key, tool_name, fn_name in mismatches:
        print(f"  Registry['{registry_key}'] = Tool(name='{tool_name}', fn='{fn_name}')")
else:
    print("No name mismatches found")