#!/usr/bin/env python3
"""
Test the final fix for the web workflow issue
"""

import asyncio
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.agents import Agent

async def test_final_fix():
    """Test the final fix for the workflow issue"""
    print("🎯 Testing Final Fix for Web Workflow Issue")
    print("=" * 50)
    
    # Load tools
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Create the exact same agent as the web workflow
    print("\n🤖 Creating agent with run_shell_command...")
    agent = Agent(
        name="agent_1",
        instructions="Execute the given shell command using the run_shell_command tool and return the exact output of the command without any additional interpretation or Zen-like response.",
        tools=["run_shell_command"],
        model="gpt-4o"
    )
    
    print("✅ Agent created successfully")
    
    # Check the agent's tools
    print("\n📋 Agent tools:")
    for i, tool in enumerate(agent.tools):
        print(f"   {i+1}. Name: {tool.name}")
        print(f"      Function: {tool.fn.__name__}")
        print(f"      Description: {tool.description[:50]}...")
        
        # Check if name matches function
        if tool.name == tool.fn.__name__:
            print("      ✅ Name and function match!")
        else:
            print(f"      ❌ Name '{tool.name}' doesn't match function '{tool.fn.__name__}'")
            return False
    
    # Test the workflow
    print("\n🚀 Testing shell command execution...")
    result = await agent.run("Execute the shell command 'ls' to list directory contents")
    
    if result and 'tool_usage_results' in result:
        tool_usage = result['tool_usage_results']
        if tool_usage:
            used_tool = tool_usage[0].tool_name
            print(f"   Tool used: {used_tool}")
            
            if used_tool == "run_shell_command":
                print("   ✅ CORRECT: Used run_shell_command!")
                print(f"   Result: {result.get('result', 'No result')[:100]}...")
                return True
            else:
                print(f"   ❌ WRONG: Used {used_tool} instead of run_shell_command")
                return False
    
    print("   ⚠️  No tool usage detected")
    return False

async def main():
    """Main test runner"""
    print("🚀 FINAL FIX TEST")
    print("=" * 50)
    print("Testing the complete fix for web workflow tool resolution")
    print()
    
    success = await test_final_fix()
    
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 FINAL FIX SUCCESSFUL!")
        print("✅ Web workflow will now use the correct tools")
        print("✅ run_shell_command works correctly")
        print("✅ No more get_analyst_recommendations confusion")
        print("\n🏆 The Bash Command Response Workflow is now FIXED!")
    else:
        print("❌ FINAL FIX FAILED")
        print("The issue persists")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())