#!/usr/bin/env python3
"""
Test dict-based tool resolution that happens when loading from YAML
This reproduces the issue that the other tests missed
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.agent_methods.agents.tool_factory import resolve_tools
from iointel.src.agent_methods.data_models.datamodels import AgentParams, Tool
from iointel.src.utilities.registries import TOOLS_REGISTRY

def test_dict_tool_resolution():
    """Test tool resolution when tools come from dicts (like YAML deserialization)"""
    print("🧪 Testing Dict-Based Tool Resolution (YAML simulation)")
    print("=" * 60)
    
    # Load tools to populate registry
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Get a real tool to simulate
    if "run_shell_command" not in TOOLS_REGISTRY:
        print("❌ run_shell_command not in registry")
        return
        
    real_tool = TOOLS_REGISTRY["run_shell_command"]
    
    # Simulate what happens when a tool is serialized to YAML
    # This is what the web workflow receives
    tool_dict = {
        "name": real_tool.name,
        "description": real_tool.description,
        "parameters": real_tool.parameters,
        "is_async": real_tool.is_async,
        "body": real_tool.body,  # Source code
        # Note: fn and fn_metadata are excluded from serialization
    }
    
    print("\n📄 Simulated YAML tool dict:")
    print(f"   name: {tool_dict['name']}")
    print(f"   has body: {tool_dict.get('body') is not None}")
    print(f"   has fn: {'fn' in tool_dict}")  # Should be False
    
    # Create AgentParams with dict tool (like from YAML)
    agent_params = AgentParams(
        name="test_agent",
        instructions="Test agent with dict tools",
        tools=[tool_dict]  # Dict tool, not string!
    )
    
    print("\n🔍 Testing resolve_tools with dict tool:")
    try:
        resolved_tools = resolve_tools(agent_params)
        print(f"   ✅ resolve_tools succeeded: {len(resolved_tools)} tools")
        
        if resolved_tools:
            tool = resolved_tools[0]
            print(f"   Tool name: {tool.name}")
            print(f"   Has fn: {tool.fn is not None}")
            print(f"   Function name: {tool.fn.__name__ if tool.fn else 'None'}")
            
            # This is where the error would happen
            if tool.fn:
                print("   ✅ Tool function properly resolved")
            else:
                print("   ❌ Tool function is None - this would cause NoneType error!")
                
    except Exception as e:
        print(f"   ❌ Error in resolve_tools: {e}")
        import traceback
        traceback.print_exc()
    
    # Also test with Tool instance (another path)
    print("\n🔍 Testing with Tool instance (no fn):")
    tool_instance = Tool.model_validate(tool_dict)
    print(f"   Tool instance fn: {tool_instance.fn}")
    print(f"   Tool instance body: {len(tool_instance.body) if tool_instance.body else 0} chars")
    
    agent_params2 = AgentParams(
        name="test_agent2",
        instructions="Test agent with Tool instance",
        tools=[tool_instance]
    )
    
    try:
        resolved_tools2 = resolve_tools(agent_params2)
        print(f"   ✅ resolve_tools succeeded: {len(resolved_tools2)} tools")
        if resolved_tools2 and resolved_tools2[0].fn:
            print("   ✅ Tool function properly resolved")
        else:
            print("   ❌ Tool function is None")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_dict_tool_resolution()