#!/usr/bin/env python3

"""
Test script to verify the improved workflow generation with better prompts and validation.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'iointel', 'src'))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.memory import AsyncMemory
from uuid import uuid4

async def test_temperature_workflow_generation():
    """Test generating a temperature workflow with multiple cities and addition."""
    print("🧪 Testing temperature workflow generation...")
    
    # Initialize memory
    memory = AsyncMemory("sqlite+aiosqlite:///test_conversations.db")
    await memory.init_models()
    
    # Create tool catalog with our available tools
    tool_catalog = {
        "get_weather": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {"city": "str"},
            "is_async": False
        },
        "add": {
            "name": "add", 
            "description": "Add two numbers",
            "parameters": {"a": "float", "b": "float"},
            "is_async": False
        }
    }
    
    # Initialize WorkflowPlanner
    planner = WorkflowPlanner(
        memory=memory,
        conversation_id=f"test_workflow_generation_{uuid4()}",
        debug=True
    )
    
    # Test query - similar to the one that failed
    query = "Get the temperature for New York and Los Angeles, then add the temperatures together"
    
    print(f"📋 Query: {query}")
    print(f"🔧 Tool catalog: {tool_catalog}")
    
    try:
        # Generate workflow with enhanced validation
        workflow = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog,
            max_retries=3
        )
        
        print(f"✅ Generated workflow: '{workflow.title}'")
        print(f"📊 Nodes: {len(workflow.nodes)}")
        print(f"🔗 Edges: {len(workflow.edges)}")
        
        # Print each node's configuration
        print("\n🔍 Node configurations:")
        for i, node in enumerate(workflow.nodes):
            print(f"  {i+1}. {node.id} ({node.type}): {node.label}")
            print(f"     Tool: {node.data.tool_name}")
            print(f"     Config: {node.data.config}")
            print(f"     Ins: {node.data.ins}")
            print(f"     Outs: {node.data.outs}")
        
        # Print edges
        print("\n🔗 Edges:")
        for i, edge in enumerate(workflow.edges):
            print(f"  {i+1}. {edge.source} -> {edge.target}")
            print(f"     Handles: {edge.sourceHandle} -> {edge.targetHandle}")
        
        # Validate the generated workflow
        issues = workflow.validate_structure(tool_catalog)
        if issues:
            print("\n❌ Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ Workflow validation passed!")
        
        # Check specifically for 'add' tool configuration
        add_nodes = [node for node in workflow.nodes if node.data.tool_name == "add"]
        if add_nodes:
            add_node = add_nodes[0]
            print("\n🔍 Add node configuration:")
            print(f"  Config: {add_node.data.config}")
            
            # Check if it has both 'a' and 'b' parameters
            if 'a' in add_node.data.config and 'b' in add_node.data.config:
                print("  ✅ Add node has both 'a' and 'b' parameters")
                
                # Check if they are data flow references
                a_val = add_node.data.config['a']
                b_val = add_node.data.config['b']
                print(f"  🔍 a = {a_val}")
                print(f"  🔍 b = {b_val}")
                
                if isinstance(a_val, str) and '{' in a_val and isinstance(b_val, str) and '{' in b_val:
                    print("  ✅ Both parameters use data flow references")
                else:
                    print("  ⚠️ Parameters don't use data flow references")
            else:
                print("  ❌ Add node missing required parameters")
        else:
            print("\n❌ No 'add' tool node found in generated workflow")
        
        print("\n🎯 Workflow reasoning:")
        print(f"  {workflow.reasoning}")
        
    except Exception as e:
        print(f"❌ Error generating workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_temperature_workflow_generation())