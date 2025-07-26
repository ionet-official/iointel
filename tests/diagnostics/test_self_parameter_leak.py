import os
#!/usr/bin/env python3
"""
Self Parameter Leak Investigation
================================
Focus on why 'self' parameters are leaking into tool catalogs
"""
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.utilities.registries import TOOLS_REGISTRY
import inspect
import os

def investigate_self_parameter_leak():
    """Investigate why 'self' appears in tool parameters."""
    print("🔍 SELF PARAMETER LEAK INVESTIGATION")
    print("=" * 60)
    
    # Load tools
    print("📋 Loading tools...")
    load_tools_from_env()
    
    # Look for tools with 'self' in their parameters
    print("\n🔍 Tools with 'self' in catalog parameters:")
    
    catalog = create_tool_catalog()
    self_parameter_tools = []
    
    for tool_name, catalog_entry in catalog.items():
        params = catalog_entry.get('parameters', {})
        if 'self' in params:
            self_parameter_tools.append(tool_name)
            
    print(f"Found {len(self_parameter_tools)} tools with 'self' parameter:")
    for tool_name in self_parameter_tools:
        print(f"  ❌ {tool_name}")
    
    if not self_parameter_tools:
        print("✅ No 'self' parameters found in catalog")
        return
    
    # Investigate the first problematic tool
    problem_tool_name = self_parameter_tools[0]
    print(f"\n🔧 Investigating: {problem_tool_name}")
    
    if problem_tool_name in TOOLS_REGISTRY:
        registry_tool = TOOLS_REGISTRY[problem_tool_name]
        catalog_entry = catalog[problem_tool_name]
        
        print(f"Registry tool type: {type(registry_tool.fn)}")
        print(f"Registry tool function: {registry_tool.fn}")
        
        # Check if it's a bound method
        if hasattr(registry_tool.fn, '__self__'):
            print(f"✅ Is bound method: {type(registry_tool.fn.__self__).__name__}")
            print(f"Method name: {registry_tool.fn.__name__}")
        else:
            print("❌ Not a bound method")
        
        # Look at the actual signature
        try:
            sig = inspect.signature(registry_tool.fn)
            print(f"Actual signature: {sig}")
            print(f"Parameters: {list(sig.parameters.keys())}")
        except Exception as e:
            print(f"Error getting signature: {e}")
        
        # Check the tool's parameters field
        print(f"Tool.parameters type: {type(registry_tool.parameters)}")
        print(f"Tool.parameters: {registry_tool.parameters}")
        
        # Check what get_wrapped_fn returns
        if hasattr(registry_tool, 'get_wrapped_fn'):
            try:
                wrapped_fn = registry_tool.get_wrapped_fn()
                wrapped_sig = inspect.signature(wrapped_fn)
                print(f"Wrapped function signature: {wrapped_sig}")
                print(f"Wrapped parameters: {list(wrapped_sig.parameters.keys())}")
            except Exception as e:
                print(f"Error with wrapped function: {e}")
        
        # Check what the catalog creation process extracted
        print(f"Catalog parameters: {catalog_entry['parameters']}")


def test_tool_creation_process():
    """Test the tool creation and catalog process step by step."""
    print("\n🛠️  TOOL CREATION PROCESS TEST")
    print("=" * 60)
    
    # Clear registry first (for clean test)
    print("🧹 Current registry size:", len(TOOLS_REGISTRY))
    
    # Import a specific tool to test
    print("📦 Testing with specific tool import...")
    
    try:
        # Import context tree to test bound method behavior
        print("✅ Imported context_tree tool")
        
        # Check what's in registry after import
        if 'read_context_tree' in TOOLS_REGISTRY:
            tool = TOOLS_REGISTRY['read_context_tree']
            print(f"Tool function: {tool.fn}")
            print(f"Tool parameters: {tool.parameters}")
            
            # Check signature
            sig = inspect.signature(tool.fn)
            print(f"Function signature: {sig}")
            
            # Try to create catalog entry for just this tool
            print("\n🔧 Testing catalog creation for this tool...")
            
            from pydantic_ai._function_schema import function_schema
            from pydantic_ai.tools import GenerateToolJsonSchema
            
            try:
                # Use the same process as create_tool_catalog
                func_schema = function_schema(
                    tool.get_wrapped_fn(),
                    schema_generator=GenerateToolJsonSchema,
                    takes_ctx=False,
                    docstring_format='auto',
                    require_parameter_descriptions=False
                )
                
                print(f"Pydantic-AI schema: {func_schema.json_schema}")
                
                json_schema = func_schema.json_schema
                properties = json_schema.get("properties", {})
                print(f"Properties from schema: {list(properties.keys())}")
                
                if 'self' in properties:
                    print("❌ FOUND THE PROBLEM: 'self' is in the pydantic-ai generated schema!")
                else:
                    print("✅ No 'self' in pydantic-ai schema")
            
            except Exception as e:
                print(f"Error with pydantic-ai schema: {e}")
                import traceback
                traceback.print_exc()
                
                # Try fallback method
                print("\n🔄 Trying fallback method...")
                if isinstance(tool.parameters, dict) and "properties" in tool.parameters:
                    properties = tool.parameters["properties"]
                    print(f"Fallback properties: {list(properties.keys())}")
        
    except Exception as e:
        print(f"Error testing specific tool: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    investigate_self_parameter_leak()
    test_tool_creation_process()