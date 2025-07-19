#!/usr/bin/env python3

"""
Test script to verify the tool catalog fix correctly extracts parameter names.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'iointel', 'src'))

from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.web.workflow_server import create_tool_catalog

def test_tool_catalog_extraction():
    """Test that the tool catalog correctly extracts parameter names."""
    print("🧪 Testing tool catalog parameter extraction...")
    
    # Import example tools to register them
    import iointel.src.RL.example_tools
    
    print(f"📋 Found {len(TOOLS_REGISTRY)} tools in registry")
    
    # Check the raw tool parameters (JSON schema)
    if "get_weather" in TOOLS_REGISTRY:
        tool = TOOLS_REGISTRY["get_weather"]
        print(f"\n🔍 Raw get_weather tool parameters:")
        print(f"  Type: {type(tool.parameters)}")
        print(f"  Content: {tool.parameters}")
        
        # Check if it's a JSON schema
        if isinstance(tool.parameters, dict) and "properties" in tool.parameters:
            properties = tool.parameters["properties"]
            print(f"  Properties keys: {list(properties.keys())}")
            for param_name, param_info in properties.items():
                print(f"    {param_name}: {param_info}")
        
    # Test the fixed catalog creation
    catalog = create_tool_catalog()
    
    print(f"\n📊 Created catalog with {len(catalog)} tools")
    
    # Check specific tools
    if "get_weather" in catalog:
        weather_tool = catalog["get_weather"]
        print(f"\n✅ get_weather tool in catalog:")
        print(f"  Name: {weather_tool['name']}")
        print(f"  Description: {weather_tool['description']}")
        print(f"  Parameters: {weather_tool['parameters']}")
        print(f"  Parameter keys: {list(weather_tool['parameters'].keys())}")
        
        # Check if it has the expected 'city' parameter
        if 'city' in weather_tool['parameters']:
            print(f"  ✅ Has 'city' parameter: {weather_tool['parameters']['city']}")
        else:
            print(f"  ❌ Missing 'city' parameter")
    
    if "add" in catalog:
        add_tool = catalog["add"]
        print(f"\n✅ add tool in catalog:")
        print(f"  Name: {add_tool['name']}")
        print(f"  Description: {add_tool['description']}")
        print(f"  Parameters: {add_tool['parameters']}")
        print(f"  Parameter keys: {list(add_tool['parameters'].keys())}")
        
        # Check if it has the expected 'a' and 'b' parameters
        if 'a' in add_tool['parameters'] and 'b' in add_tool['parameters']:
            print(f"  ✅ Has 'a' and 'b' parameters: a={add_tool['parameters']['a']}, b={add_tool['parameters']['b']}")
        else:
            print(f"  ❌ Missing 'a' or 'b' parameters")
    
    # Check for the problematic schema fields
    problematic_fields = ['properties', 'required', 'title', 'type']
    print(f"\n🔍 Checking for problematic schema fields...")
    for tool_name, tool_info in catalog.items():
        tool_params = tool_info['parameters']
        found_problematic = [field for field in problematic_fields if field in tool_params]
        if found_problematic:
            print(f"  ❌ {tool_name} still has schema fields: {found_problematic}")
        else:
            print(f"  ✅ {tool_name} has clean parameters: {list(tool_params.keys())}")
    
    print(f"\n🎯 Tool catalog should now work correctly with validation!")

if __name__ == "__main__":
    test_tool_catalog_extraction()