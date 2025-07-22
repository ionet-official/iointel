#!/usr/bin/env python3
"""
Tool Instance Diagnosis Test
============================
Investigate the 'self' error by examining tool registration vs execution instances.
"""
import sys
sys.path.append('/Users/alexandermorisse/Documents/GitHub/iointel')

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.agents import Agent
import inspect

def analyze_tool_instances():
    """Analyze tool registration and instance relationships."""
    print("🔍 TOOL INSTANCE DIAGNOSIS")
    print("=" * 60)
    
    # Step 1: Load tools and see what gets registered
    print("📋 Step 1: Loading tools from environment...")
    available_tools = load_tools_from_env()
    print(f"✅ Loaded {len(available_tools)} tools from environment")
    print(f"✅ TOOLS_REGISTRY now contains {len(TOOLS_REGISTRY)} tools")
    
    # Step 2: Analyze a few key tools that might have bound methods
    print("\n📋 Step 2: Analyzing bound method tools...")
    
    bound_method_tools = []
    function_tools = []
    
    for tool_name, tool in list(TOOLS_REGISTRY.items())[:10]:  # Check first 10
        if hasattr(tool.fn, '__self__'):
            bound_method_tools.append((tool_name, tool))
            print(f"🔗 BOUND: {tool_name} -> {type(tool.fn.__self__).__name__}")
            print(f"   Instance ID: {id(tool.fn.__self__)}")
            print(f"   Method: {tool.fn.__name__}")
        else:
            function_tools.append((tool_name, tool))
            print(f"📦 FUNC: {tool_name} -> {tool.fn.__name__}")
    
    print(f"\n📊 Summary: {len(bound_method_tools)} bound methods, {len(function_tools)} functions")
    
    # Step 3: Create tool catalog and see if instances are preserved
    print("\n📋 Step 3: Creating tool catalog...")
    try:
        catalog = create_tool_catalog()
        print(f"✅ Tool catalog created with {len(catalog)} entries")
    except Exception as e:
        print(f"❌ Tool catalog creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test if we can actually call a bound method tool
    print("\n📋 Step 4: Testing bound method tool execution...")
    
    if bound_method_tools:
        tool_name, tool = bound_method_tools[0]
        print(f"🧪 Testing: {tool_name}")
        
        try:
            # Get the function signature to see what parameters it expects
            sig = inspect.signature(tool.fn)
            print(f"   Signature: {sig}")
            
            # Try to get the wrapped function (if it exists)
            if hasattr(tool, 'get_wrapped_fn'):
                wrapped_fn = tool.get_wrapped_fn()
                print(f"   Wrapped function: {wrapped_fn}")
                print(f"   Wrapped function self: {getattr(wrapped_fn, '__self__', 'No self')}")
                
                # Compare instance IDs
                if hasattr(tool.fn, '__self__') and hasattr(wrapped_fn, '__self__'):
                    original_id = id(tool.fn.__self__)
                    wrapped_id = id(wrapped_fn.__self__)
                    print(f"   Original instance ID: {original_id}")
                    print(f"   Wrapped instance ID: {wrapped_id}")
                    print(f"   IDs match: {original_id == wrapped_id}")
            
        except Exception as e:
            print(f"   ❌ Error testing tool: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Test agent creation with these tools
    print("\n📋 Step 5: Testing Agent creation with bound method tools...")
    
    if bound_method_tools:
        test_tool_names = [name for name, _ in bound_method_tools[:3]]
        print(f"🧪 Testing agent with tools: {test_tool_names}")
        
        try:
            agent_params = AgentParams(
                name="test_agent",
                instructions="Test agent for diagnosing tool instances",
                tools=test_tool_names
            )
            
            print("✅ AgentParams created successfully")
            print(f"   Tools: {agent_params.tools}")
            
            # Try to create the actual Agent
            agent = Agent(**agent_params.model_dump(exclude={"tools"}), tools=agent_params.tools)
            print("✅ Agent created successfully")
            print(f"   Agent tools count: {len(agent.tools) if hasattr(agent, 'tools') else 'No tools attr'}")
            
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            print("   This might be our 'self' error!")
            import traceback
            traceback.print_exc()
    
    # Step 6: Investigate tool resolution process
    print("\n📋 Step 6: Testing tool resolution...")
    
    if bound_method_tools:
        tool_name, original_tool = bound_method_tools[0]
        print(f"🧪 Testing tool resolution for: {tool_name}")
        
        from iointel.src.utilities.tool_registry_utils import resolve_tool
        
        try:
            resolved_tool = resolve_tool(tool_name)
            print("✅ Tool resolved successfully")
            
            # Compare instances
            if hasattr(original_tool.fn, '__self__') and hasattr(resolved_tool.fn, '__self__'):
                original_id = id(original_tool.fn.__self__)
                resolved_id = id(resolved_tool.fn.__self__)
                print(f"   Original instance ID: {original_id}")
                print(f"   Resolved instance ID: {resolved_id}")
                print(f"   Instances match: {original_id == resolved_id}")
                
                if original_id != resolved_id:
                    print("   ❌ INSTANCE MISMATCH DETECTED!")
                    print("   This could be the source of 'self' errors!")
            
        except Exception as e:
            print(f"❌ Tool resolution failed: {e}")
            import traceback
            traceback.print_exc()


def test_workflow_tool_usage():
    """Test how tools are used in actual workflow context."""
    print("\n🚀 WORKFLOW CONTEXT TEST")
    print("=" * 60)
    
    from iointel.src.utilities.workflow_helpers import create_tool_catalog
    
    # Load tools first
    load_tools_from_env()
    
    # Create tool catalog
    try:
        catalog = create_tool_catalog()
        print(f"✅ Created catalog with {len(catalog)} tools")
        
        # Check a few tools in the catalog
        sample_tools = list(catalog.keys())[:5]
        print(f"📋 Sample catalog tools: {sample_tools}")
        
        for tool_name in sample_tools:
            if tool_name in TOOLS_REGISTRY:
                registry_tool = TOOLS_REGISTRY[tool_name]
                catalog_entry = catalog[tool_name]
                
                print(f"🔍 {tool_name}:")
                print(f"   Registry: {type(registry_tool.fn)}")
                print(f"   Catalog params: {list(catalog_entry.get('parameters', {}).keys())}")
                
                # Check if this tool has a 'self' reference
                if hasattr(registry_tool.fn, '__self__'):
                    print(f"   Has bound instance: {type(registry_tool.fn.__self__).__name__}")
                    print(f"   Instance still alive: {registry_tool.fn.__self__ is not None}")
        
    except Exception as e:
        print(f"❌ Tool catalog creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_tool_instances()
    test_workflow_tool_usage()