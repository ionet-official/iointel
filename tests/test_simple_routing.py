#!/usr/bin/env python
"""
Simple test of routing_gate with agent.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iointel.src.agents import Agent
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env, get_tool_by_name
from iointel.src.agent_methods.tools.conditional_gate import routing_gate  # Import to register

async def test_simple():
    # Load tools
    load_tools_from_env()
    
    # Check routing_gate is available
    tool = get_tool_by_name("routing_gate")
    if tool:
        print(f"✅ routing_gate found: {tool}")
    else:
        print("❌ routing_gate NOT found!")
        return
    
    # Create simple agent
    agent = Agent(
        name="TestRouter",
        instructions="Route 'pwd' to index 4. Call: routing_gate(data='pwd', route_index=4, route_name='Shell')",
        model="gpt-4o",
        tools=["routing_gate"]
    )
    
    print("\nTesting agent with 'pwd'...")
    result = await agent.run("pwd")
    
    print(f"Result keys: {result.keys()}")
    print(f"Content: {result.get('content', 'No content')}")
    
    # Check tool_usage_results instead of tool_usage
    if 'tool_usage_results' in result:
        print(f"\nTool usage results: {result['tool_usage_results']}")
        for tool_result in result['tool_usage_results']:
            print(f"  Tool: {tool_result.tool_name if hasattr(tool_result, 'tool_name') else tool_result}")
            print(f"  Result: {tool_result.tool_result if hasattr(tool_result, 'tool_result') else 'N/A'}")
    
    if 'tool_usage' in result:
        print(f"\nTool usage: {result['tool_usage']}")
    else:
        print("\nNo tool_usage key!")

if __name__ == "__main__":
    asyncio.run(test_simple())