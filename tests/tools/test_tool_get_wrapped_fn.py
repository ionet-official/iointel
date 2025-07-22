#!/usr/bin/env python3
"""
Test the Tool.get_wrapped_fn() method to understand the KeyError: 'self' issue
"""

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY
import inspect

def test_tool_get_wrapped_fn():
    """Test get_wrapped_fn() method on agno tools"""
    print("🧪 Testing Tool.get_wrapped_fn() method")
    print("=" * 50)
    
    # Load tools
    tools = load_tools_from_env()
    print(f"✅ Loaded {len(tools)} tools")
    
    # Test specific agno tools that were problematic
    test_tools = ["run_shell_command", "arxiv_search"]
    
    for tool_name in test_tools:
        if tool_name not in TOOLS_REGISTRY:
            print(f"❌ Tool {tool_name} not found in registry")
            continue
            
        tool = TOOLS_REGISTRY[tool_name]
        print(f"\n🔍 Testing {tool_name}:")
        print(f"   Tool name: {tool.name}")
        print(f"   Function name: {tool.fn.__name__}")
        print(f"   Is method: {inspect.ismethod(tool.fn)}")
        
        # Check original function signature
        try:
            original_sig = inspect.signature(tool.fn)
            original_params = list(original_sig.parameters.keys())
            print(f"   Original signature params: {original_params}")
            has_self_in_original = 'self' in original_params
            print(f"   Original has 'self': {has_self_in_original}")
        except Exception as e:
            print(f"   ❌ Error getting original signature: {e}")
            continue
        
        # Test get_wrapped_fn()
        try:
            wrapped_fn = tool.get_wrapped_fn()
            wrapped_sig = inspect.signature(wrapped_fn)
            wrapped_params = list(wrapped_sig.parameters.keys())
            print(f"   Wrapped signature params: {wrapped_params}")
            has_self_in_wrapped = 'self' in wrapped_params
            print(f"   Wrapped has 'self': {has_self_in_wrapped}")
            
            if has_self_in_wrapped:
                print("   ❌ PROBLEM: get_wrapped_fn() still has 'self' parameter!")
            else:
                print("   ✅ get_wrapped_fn() correctly removed 'self' parameter")
                
        except Exception as e:
            print(f"   ❌ Error calling get_wrapped_fn(): {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_tool_get_wrapped_fn()