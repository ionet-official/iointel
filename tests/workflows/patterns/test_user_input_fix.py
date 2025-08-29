#!/usr/bin/env python3
"""Test that user inputs flow correctly through the system."""

import asyncio
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, NodeData
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState


async def test_user_input_flow():
    """Test that user inputs are properly passed to data source nodes."""
    
    # Create a simple workflow with a user_input data source
    import uuid
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Test User Input",
        description="Test workflow for user input",
        reasoning="Test workflow",
        nodes=[
            NodeSpec(
                id="user_input_1",
                type="data_source",
                label="Stock Symbol Input",
                data=NodeData(
                    source_name="user_input",
                    config={
                        "message": "Enter stock symbol",
                        "default_value": "AAPL"
                    },
                    outs=["symbol"]
                )
            )
        ],
        edges=[]
    )
    
    # Create DAG executor with typed execution
    dag_executor = DAGExecutor(use_typed_execution=True)
    
    # Build execution graph
    dag_executor.build_execution_graph(
        workflow_spec=workflow,
        objective="Test user input",
        conversation_id="test-conv-123"
    )
    
    # Test 1: With user inputs
    print("\n=== Test 1: With user inputs ===")
    initial_state = WorkflowState(
        initial_text="Test",
        conversation_id="test-conv-123",
        user_inputs={"any_key": "Oil ETFs"}  # User provided input
    )
    
    final_state = await dag_executor.execute_dag(initial_state)
    print(f"Final state: {final_state}")
    result = final_state.results.get("user_input_1")
    print(f"Result with user input: {result}")
    assert result == "Oil ETFs", f"Expected 'Oil ETFs' but got {result}"
    
    # Test 2: Without user inputs (should use default)
    print("\n=== Test 2: Without user inputs ===")
    initial_state2 = WorkflowState(
        initial_text="Test",
        conversation_id="test-conv-123",
        user_inputs={}  # No user input
    )
    
    final_state2 = await dag_executor.execute_dag(initial_state2)
    print(f"Final state 2: {final_state2}")
    result2 = final_state2.results.get("user_input_1")
    print(f"Result without user input: {result2}")
    assert result2 == "AAPL", f"Expected 'AAPL' but got {result2}"
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_user_input_flow())