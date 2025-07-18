#!/usr/bin/env python3
"""
Test the web workflow execution fix for KeyError: 'self'
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.agent_methods.agents.tool_factory import resolve_tools
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.utilities.registries import TOOLS_REGISTRY
import inspect

def test_web_workflow_fix():
    """Test that web workflow tool resolution works correctly"""
    print("🧪 Testing Web Workflow Tool Resolution Fix")
    print("=" * 50)
    
    # Load tools to populate registry
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Test specific agno tools that were problematic
    test_tools = ["run_shell_command", "arxiv_search"]
    
    for tool_name in test_tools:
        if tool_name not in TOOLS_REGISTRY:
            print(f"❌ Tool {tool_name} not found in registry")
            continue
            
        print(f"\n🔍 Testing {tool_name}:")
        
        # Create AgentParams with the tool (simulating web workflow)
        agent_params = AgentParams(
            name="test_agent",
            instructions="Test agent",
            tools=[tool_name]  # String tool lookup like web workflow
        )
        
        try:
            # This simulates what happens in web workflow execution
            resolved_tools = resolve_tools(agent_params)
            print(f"   ✅ resolve_tools succeeded: {len(resolved_tools)} tools")
            
            # Check the resolved tool
            if resolved_tools:
                tool = resolved_tools[0]
                print(f"   Tool name: {tool.name}")
                print(f"   Function name: {tool.fn.__name__}")
                print(f"   Is method: {inspect.ismethod(tool.fn)}")
                
                # Check get_wrapped_fn() - this is what caused the error
                try:
                    wrapped_fn = tool.get_wrapped_fn()
                    wrapped_sig = inspect.signature(wrapped_fn)
                    wrapped_params = list(wrapped_sig.parameters.keys())
                    print(f"   Wrapped signature params: {wrapped_params}")
                    has_self_in_wrapped = 'self' in wrapped_params
                    print(f"   Wrapped has 'self': {has_self_in_wrapped}")
                    
                    if has_self_in_wrapped:
                        print(f"   ❌ PROBLEM: get_wrapped_fn() still has 'self' parameter!")
                    else:
                        print(f"   ✅ get_wrapped_fn() correctly removed 'self' parameter")
                        
                except Exception as e:
                    print(f"   ❌ Error calling get_wrapped_fn(): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ❌ No tools resolved")
                
        except Exception as e:
            print(f"   ❌ Error in resolve_tools: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏆 Web workflow fix test completed!")

if __name__ == "__main__":
    test_web_workflow_fix()