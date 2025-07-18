#!/usr/bin/env python3
"""
Test agno tools with new auto-registration pattern (no load_tools_from_env)
"""

# Import agno tools to trigger registration
from iointel.src.agent_methods.tools.agno.shell import Shell
from iointel.src.agent_methods.tools.agno.arxiv import Arxiv

from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.agents import Agent
import inspect

def test_agno_new_pattern():
    """Test agno tools with new auto-registration pattern"""
    print("🧪 Testing Agno Tools with New Pattern")
    print("=" * 50)
    
    # Create instances to trigger registration
    shell = Shell()
    arxiv = Arxiv()
    
    print(f"✅ Created Shell instance: {shell}")
    print(f"✅ Created Arxiv instance: {arxiv}")
    
    # Check registry
    print(f"\n📊 Registry status:")
    print(f"   Total tools: {len(TOOLS_REGISTRY)}")
    
    # Check our fixed agno tools
    test_tools = ["run_shell_command", "arxiv_search"]
    
    for tool_name in test_tools:
        if tool_name in TOOLS_REGISTRY:
            tool = TOOLS_REGISTRY[tool_name]
            print(f"   ✅ {tool_name}: Found")
            print(f"      Has fn: {tool.fn is not None}")
            
            # Test our KeyError: 'self' fix
            if tool.fn:
                try:
                    wrapped_fn = tool.get_wrapped_fn()
                    sig = inspect.signature(wrapped_fn)
                    params = list(sig.parameters.keys())
                    has_self = 'self' in params
                    print(f"      Wrapped params: {params}")
                    print(f"      Has 'self': {has_self}")
                    
                    if has_self:
                        print(f"      ❌ Still has 'self' - fix failed!")
                    else:
                        print(f"      ✅ No 'self' - fix working!")
                        
                except Exception as e:
                    print(f"      ❌ Error getting wrapped fn: {e}")
        else:
            print(f"   ❌ {tool_name}: Not found")
    
    # Test agent creation (this was failing before)
    print(f"\n🤖 Testing Agent creation:")
    try:
        agent = Agent(
            name="test_agent",
            instructions="Test agent",
            tools=["run_shell_command"]
        )
        print(f"   ✅ Agent created successfully: {len(agent.tools)} tools")
        print(f"   Agent tools: {[t.name for t in agent.tools]}")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agno_new_pattern()