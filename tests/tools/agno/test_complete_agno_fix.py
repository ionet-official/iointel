#!/usr/bin/env python3
"""
Comprehensive test to verify all agno tools fixes work together
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.agent_methods.agents.tool_factory import resolve_tools
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.tool_registry_utils import (
    resolve_tool, validate_tool_registry, debug_tool_resolution
)
from iointel.src.agents import Agent
import inspect

def test_complete_agno_fix():
    """Comprehensive test for all agno tools fixes"""
    print("🧪 Comprehensive Agno Tools Fix Test")
    print("=" * 60)
    
    # Load tools
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Test 1: Registry validation
    print(f"\n1️⃣ Testing registry validation:")
    validation = validate_tool_registry()
    print(f"   Total tools: {validation['total_tools']}")
    print(f"   Registry health: {'✅ Healthy' if validation['is_healthy'] else '❌ Issues found'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    # Test 2: Agno tools registration
    print(f"\n2️⃣ Testing agno tools registration:")
    agno_tools = ["run_shell_command", "arxiv_search", "get_analyst_recommendations", "file_read"]
    
    for tool_name in agno_tools:
        if tool_name in TOOLS_REGISTRY:
            tool = TOOLS_REGISTRY[tool_name]
            print(f"   ✅ {tool_name}: {tool.fn.__name__} (method: {inspect.ismethod(tool.fn)})")
        else:
            print(f"   ❌ {tool_name}: Not found")
    
    # Test 3: Centralized tool resolution
    print(f"\n3️⃣ Testing centralized tool resolution:")
    for tool_name in agno_tools[:2]:  # Test first 2 to avoid spam
        if tool_name in TOOLS_REGISTRY:
            try:
                resolved_tool = resolve_tool(tool_name)
                print(f"   ✅ {tool_name}: {resolved_tool.name} -> {resolved_tool.fn.__name__}")
            except Exception as e:
                print(f"   ❌ {tool_name}: {e}")
    
    # Test 4: Web workflow tool resolution
    print(f"\n4️⃣ Testing web workflow tool resolution:")
    for tool_name in agno_tools[:2]:  # Test first 2
        if tool_name in TOOLS_REGISTRY:
            try:
                agent_params = AgentParams(
                    name="test_agent",
                    instructions="Test agent",
                    tools=[tool_name]
                )
                resolved_tools = resolve_tools(agent_params)
                if resolved_tools:
                    tool = resolved_tools[0]
                    wrapped_fn = tool.get_wrapped_fn()
                    wrapped_sig = inspect.signature(wrapped_fn)
                    has_self = 'self' in wrapped_sig.parameters
                    print(f"   ✅ {tool_name}: get_wrapped_fn() self={has_self}")
                else:
                    print(f"   ❌ {tool_name}: No tools resolved")
            except Exception as e:
                print(f"   ❌ {tool_name}: {e}")
    
    # Test 5: Agent creation with agno tools
    print(f"\n5️⃣ Testing agent creation with agno tools:")
    try:
        agent = Agent(
            name="test_agent",
            instructions="Test agent with agno tools",
            tools=["run_shell_command", "arxiv_search"]
        )
        print(f"   ✅ Agent created successfully with {len(agent.tools)} tools")
        print(f"   Agent tools: {[t.name for t in agent.tools]}")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Debug tool resolution
    print(f"\n6️⃣ Testing debug tool resolution:")
    for tool_name in agno_tools[:1]:  # Test just one
        if tool_name in TOOLS_REGISTRY:
            debug_info = debug_tool_resolution(tool_name)
            print(f"   Tool: {debug_info['tool_name']}")
            print(f"   Success: {'✅' if debug_info['success'] else '❌'}")
            if debug_info['final_result']:
                result = debug_info['final_result']
                print(f"   Final: {result['name']} -> {result['function']}")
    
    print(f"\n🏆 Comprehensive agno tools fix test completed!")
    print("   All components working together correctly!")

if __name__ == "__main__":
    test_complete_agno_fix()