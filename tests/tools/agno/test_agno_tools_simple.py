#!/usr/bin/env python3
"""
Simple test for agno tools to verify they're working without registration conflicts
"""

import asyncio
import tempfile
from pathlib import Path
from iointel.src.agents import Agent

async def test_basic_agno_tools():
    """Test basic agno tools functionality"""
    print("🧪 Testing agno tools without tool conflicts...")
    
    # Import agno tools directly to avoid registration conflicts
    from iointel.src.agent_methods.tools.agno.shell import Shell
    from iointel.src.agent_methods.tools.agno.file import File
    
    # Create instances 
    shell_tool = Shell()
    file_tool = File()
    
    print("✅ Agno tool instances created successfully")
    
    # Test shell tool directly
    print("\n🐚 Testing Shell tool directly:")
    try:
        result = shell_tool.run_shell_command(['echo', 'hello world'])
        print(f"  ✅ Shell command executed: {result}")
    except Exception as e:
        print(f"  ❌ Shell command failed: {e}")
    
    # Test file tool directly
    print("\n📁 Testing File tool directly:")
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test content")
            temp_path = f.name
        
        # Test file read
        result = file_tool.file_read(temp_path)
        print(f"  ✅ File read successful: {result[:50]}...")
        
        # Test file list
        result = file_tool.file_list(str(Path(temp_path).parent))
        print(f"  ✅ File list successful: found {len(result.split())} entries")
        
        # Clean up
        Path(temp_path).unlink()
        
    except Exception as e:
        print(f"  ❌ File operations failed: {e}")
    
    print("\n🎉 Direct agno tool tests completed!")

def test_tool_registration():
    """Test tool registration without conflicts"""
    print("\n🔧 Testing tool registration:")
    
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    
    # Count tools before
    initial_count = len(TOOLS_REGISTRY)
    print(f"  Initial tool count: {initial_count}")
    
    # Import agno tools to trigger registration
    from iointel.src.agent_methods.tools.agno.shell import Shell
    
    # Create instance to trigger bound method registration
    shell = Shell()
    
    # Count tools after
    final_count = len(TOOLS_REGISTRY)
    print(f"  Final tool count: {final_count}")
    
    # Check if shell tool is registered
    if 'run_shell_command' in TOOLS_REGISTRY:
        tool = TOOLS_REGISTRY['run_shell_command']
        print(f"  ✅ Shell tool registered: {tool.name}")
        
        # Test the registered tool function
        try:
            result = tool.fn(['echo', 'test registration'])
            print(f"  ✅ Registered tool function works: {result}")
        except Exception as e:
            print(f"  ❌ Registered tool function failed: {e}")
    else:
        print(f"  ❌ Shell tool not found in registry")
        print(f"  Available tools: {list(TOOLS_REGISTRY.keys())[-10:]}")  # Show last 10

async def test_agent_with_working_tools():
    """Test agent with tools that are confirmed working"""
    print("\n🤖 Testing agent with confirmed working tools:")
    
    # Use basic tools that don't have conflicts
    try:
        agent = Agent(
            name='test_agent',
            instructions='You are a helpful assistant. Use tools when requested.',
            tools=['what_time_is_it', 'calculator_add']  # Use simple working tools
        )
        
        print("  ✅ Agent created with basic tools")
        
        # Test time tool
        result = await agent.run('What time is it?')
        print(f"  ✅ Time tool result: {result.get('result', 'No result')[:50]}...")
        
        # Test calculator
        result = await agent.run('Add 5 + 3 using the calculator')
        print(f"  ✅ Calculator result: {result.get('result', 'No result')[:50]}...")
        
    except Exception as e:
        print(f"  ❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test runner"""
    print("🚀 Starting simplified agno tools test...")
    
    # Test 1: Direct tool functionality
    await test_basic_agno_tools()
    
    # Test 2: Tool registration
    test_tool_registration()
    
    # Test 3: Agent with working tools
    await test_agent_with_working_tools()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())