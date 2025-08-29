#!/usr/bin/env python3
"""Test user inputs work correctly with plan_and_execute (mimics web app flow)."""

import asyncio
from iointel.src.utilities.workflow_helpers import generate_only, plan_and_execute
from iointel.src.utilities.graph_nodes import WorkflowState


async def test_user_input_with_workflow_helpers():
    """Test that user inputs flow correctly through the workflow helpers."""
    
    # Test 1: Generate a workflow that uses user input
    print("\n=== Test 1: Generate workflow with user input ===")
    
    workflow = await generate_only("Create a simple stock analysis workflow")
    
    if not workflow:
        print("❌ Failed to generate workflow")
        return
        
    print(f"✅ Generated workflow: {workflow.title}")
    print(f"   Nodes: {len(workflow.nodes)}")
    print(f"   Edges: {len(workflow.edges)}")
    
    # Check if we have a user_input node
    user_input_nodes = [n for n in workflow.nodes if n.type == "data_source" and n.data.source_name == "user_input"]
    if user_input_nodes:
        print(f"✅ Found {len(user_input_nodes)} user_input nodes:")
        for node in user_input_nodes:
            print(f"   - {node.label}: {node.data.config.get('message', 'N/A')}")
            print(f"     Default: {node.data.config.get('default_value', 'N/A')}")
    else:
        print("⚠️  No user_input nodes found in generated workflow")
    
    # Test 2: Execute with user inputs using plan_and_execute
    print("\n=== Test 2: Execute with user inputs (plan_and_execute) ===")
    
    # Create initial state with user inputs
    initial_state = WorkflowState(
        initial_text="Test with user input",
        conversation_id="test-123",
        user_inputs={"stock_key": "NVDA"}  # User wants NVDA, not default AAPL
    )
    
    result = await plan_and_execute(
        "Analyze the stock and give me a recommendation",
        initial_state=initial_state
    )
    
    if result["success"]:
        print("✅ Workflow executed successfully!")
        final_state = result["execution_result"]
        
        # Check if user input was used
        for node_id, node_result in final_state.results.items():
            print(f"\n   Node {node_id}: {node_result}")
            
            # Look for evidence that NVDA was used instead of default
            if isinstance(node_result, str) and "NVDA" in node_result:
                print("   ✅ Found user input 'NVDA' in results!")
            elif isinstance(node_result, str) and "AAPL" in node_result:
                print("   ⚠️  Found default 'AAPL' - user input might not have been used")
                
    else:
        print(f"❌ Workflow execution failed: {result.get('error', 'Unknown error')}")
    
    # Test 3: Execute without user inputs (should use defaults)
    print("\n=== Test 3: Execute without user inputs (should use defaults) ===")
    
    result2 = await plan_and_execute(
        "Analyze the stock and give me a recommendation"
        # No initial_state, so no user_inputs
    )
    
    if result2["success"]:
        print("✅ Workflow executed successfully!")
        final_state2 = result2["execution_result"]
        
        # Check if default was used
        for node_id, node_result in final_state2.results.items():
            if isinstance(node_result, str):
                print(f"\n   Node {node_id}: {node_result[:100]}...")
                
    else:
        print(f"❌ Workflow execution failed: {result2.get('error', 'Unknown error')}")


async def test_specific_user_input_workflow():
    """Test a specific workflow that definitely uses user input."""
    
    print("\n\n=== Test 4: Specific user input workflow ===")
    
    # Create a workflow that asks for user input
    initial_state = WorkflowState(
        initial_text="What ETFs should I analyze?",
        conversation_id="test-456", 
        user_inputs={"user_query": "I have oil ETFs from 2020, how are they doing?"}
    )
    
    result = await plan_and_execute(
        "Ask the user what they want to analyze and provide insights",
        initial_state=initial_state
    )
    
    if result["success"]:
        print("✅ Workflow executed successfully!")
        
        # Analyze the workflow that was generated
        workflow = result["workflow_spec"]
        print(f"\nGenerated workflow: {workflow.title}")
        
        # Look for user input nodes
        for node in workflow.nodes:
            if node.type == "data_source" and node.data.source_name == "user_input":
                print(f"\n📝 User input node: {node.label}")
                print(f"   Config: {node.data.config}")
                
        # Check final results
        final_state = result["execution_result"]
        print("\n📊 Results:")
        for node_id, result_value in final_state.results.items():
            if "oil" in str(result_value).lower() or "etf" in str(result_value).lower():
                print(f"   ✅ Node {node_id} processed user input about oil ETFs")
                print(f"      Result preview: {str(result_value)[:200]}...")
                
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(test_user_input_with_workflow_helpers())
    asyncio.run(test_specific_user_input_workflow())