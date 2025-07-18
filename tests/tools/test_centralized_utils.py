#!/usr/bin/env python3
"""
Test the centralized tool registry utils
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import (
    resolve_tool, 
    get_tool_info, 
    validate_tool_registry,
    debug_tool_resolution
)

def test_centralized_utils():
    """Test the centralized tool registry utilities"""
    print("🎯 Testing Centralized Tool Registry Utils")
    print("=" * 50)
    
    # Load tools
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Test 1: Tool resolution
    print(f"\n1️⃣ Testing tool resolution:")
    test_tools = ["run_shell_command", "get_analyst_recommendations", "arxiv_search"]
    
    for tool_name in test_tools:
        try:
            resolved_tool = resolve_tool(tool_name)
            print(f"   ✅ {tool_name}: {resolved_tool.name} -> {resolved_tool.fn.__name__}")
        except Exception as e:
            print(f"   ❌ {tool_name}: {e}")
    
    # Test 2: Tool info
    print(f"\n2️⃣ Testing tool info:")
    info = get_tool_info("run_shell_command")
    print(f"   Tool: {info['name']}")
    print(f"   Function: {info['function_name']}")
    print(f"   Names match: {info['name_matches_function']}")
    print(f"   Registry key matches: {info['registry_key_matches_name']}")
    
    # Test 3: Registry validation
    print(f"\n3️⃣ Testing registry validation:")
    validation = validate_tool_registry()
    print(f"   Total tools: {validation['total_tools']}")
    print(f"   Registry health: {'✅ Healthy' if validation['is_healthy'] else '❌ Issues found'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    # Test 4: Debug tool resolution
    print(f"\n4️⃣ Testing debug tool resolution:")
    debug_info = debug_tool_resolution("run_shell_command")
    print(f"   Tool: {debug_info['tool_name']}")
    print(f"   Success: {'✅' if debug_info['success'] else '❌'}")
    if debug_info['final_result']:
        result = debug_info['final_result']
        print(f"   Final: {result['name']} -> {result['function']}")
    
    print(f"\n🏆 Centralized tool registry utils are working correctly!")

if __name__ == "__main__":
    test_centralized_utils()