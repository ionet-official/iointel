#!/usr/bin/env python3
"""
Test WorkflowPlanner's ability to make targeted updates with improved context.
This test creates a workflow, then updates it to change a prompt_tool to user_input.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY

# Import tools to ensure registration
import iointel.src.agent_methods.tools.conditional_gate
import iointel.src.agent_methods.tools.user_input
import iointel.src.agent_methods.tools.prompt_tool

async def test_workflow_updates():
    """Test two-turn sequence: create workflow, then update prompt to user_input."""
    
    # Load tools
    load_tools_from_env()
    print(f"✅ Loaded {len(TOOLS_REGISTRY)} tools")
    
    # Create workflow planner
    planner = WorkflowPlanner(debug=True)
    
    # Build tool catalog from registry
    tool_catalog = {}
    for tool_name, tool_info in TOOLS_REGISTRY.items():
        if hasattr(tool_info, 'get_tool_schema'):
            schema = tool_info.get_tool_schema()
            tool_catalog[tool_name] = {
                'name': tool_name,
                'description': schema.get('description', 'No description'),
                'parameters': schema.get('parameters', {}),
                'required_parameters': schema.get('required', [])
            }
        else:
            # Fallback for tools without schema
            tool_catalog[tool_name] = {
                'name': tool_name,
                'description': getattr(tool_info, '__doc__', 'No description'),
                'parameters': {},
                'required_parameters': []
            }
    
    print("\n🔄 TURN 1: Creating initial workflow with prompt_tool")
    print("=" * 60)
    
    # First query - create workflow with prompt tool
    query1 = """
    Create a simple email categorization workflow:
    1. Start with a prompt tool that provides 10 sample emails
    2. An agent that categorizes emails into: taxes, social media, house-life
    3. Use conditional_gate to route to appropriate agents based on category
    4. Three agents to handle each category
    """
    
    try:
        workflow1, _ = await planner.generate_workflow(
            query=query1,
            tool_catalog=tool_catalog
        )
        
        print(f"\n✅ Created workflow: {workflow1.title}")
        print(f"Nodes: {len(workflow1.nodes)}")
        for node in workflow1.nodes:
            if node.type == "tool":
                print(f"  - {node.id}: {node.data.tool_name}")
            else:
                print(f"  - {node.id}: {node.type}")
                
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        return
    
    print("\n🔄 TURN 2: Updating workflow to use user_input instead of prompt")
    print("=" * 60)
    
    # Second query - update to use user_input
    query2 = """
    Instead of the prompt tool with sample emails, change it to a user_input tool 
    so users can input their own email for categorization. Keep everything else the same.
    """
    
    try:
        workflow2, _ = await planner.generate_workflow(
            query=query2,
            tool_catalog=tool_catalog
        )
        
        print(f"\n✅ Updated workflow: {workflow2.title}")
        print(f"Nodes: {len(workflow2.nodes)}")
        
        # Compare the workflows
        print("\n📊 COMPARISON:")
        print("-" * 40)
        
        # Check if structure is preserved
        nodes1_ids = {n.id for n in workflow1.nodes}
        nodes2_ids = {n.id for n in workflow2.nodes}
        
        # Find the changed node
        for node in workflow2.nodes:
            if node.type == "tool":
                matching_node1 = next((n for n in workflow1.nodes if n.id == node.id), None)
                if matching_node1 and matching_node1.data.tool_name != node.data.tool_name:
                    print(f"✅ Changed: {node.id}")
                    print(f"   Before: {matching_node1.data.tool_name}")
                    print(f"   After: {node.data.tool_name}")
                elif not matching_node1:
                    print(f"🆕 New node: {node.id} ({node.data.tool_name})")
        
        # Check preserved nodes
        preserved = nodes1_ids & nodes2_ids
        if preserved:
            print(f"\n✅ Preserved {len(preserved)} nodes:")
            for node_id in preserved:
                print(f"   - {node_id}")
        
        # Check edges
        edges1_count = len(workflow1.edges)
        edges2_count = len(workflow2.edges)
        print(f"\n🔗 Edges: {edges1_count} → {edges2_count}")
        
        # Success criteria
        if len(workflow2.nodes) >= len(workflow1.nodes) - 1:  # Allow some flexibility
            print("\n✅ SUCCESS: Workflow structure largely preserved!")
        else:
            print("\n⚠️  WARNING: Significant structural changes detected")
            
    except Exception as e:
        print(f"❌ Error updating workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow_updates())