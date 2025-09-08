#!/usr/bin/env python3
"""
Test DAG execution flow with user_input data source
"""

import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
if env_path.exists():
    load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.utilities.dag_executor import create_dag_executor_from_spec
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.io_logger import get_component_logger

logger = get_component_logger("TEST_DAG_FLOW")


def create_simple_test_workflow():
    """Create a minimal user_input -> agent workflow."""
    
    nodes = [
        # User input node
        NodeSpec(
            id="user_input_1",
            type="data_source",
            label="User Input",
            data=NodeData(
                source_name="user_input",
                config={
                    "message": "What's your question?",
                    "default_value": "Hello, I need help!"
                },
                ins=[],
                outs=["user_query"]
            )
        ),
        
        # Simple agent that should process the user input
        NodeSpec(
            id="agent_1",
            type="agent",
            label="Chat Agent",
            data=NodeData(
                agent_instructions="You are a helpful assistant. Process this user request: {user_input_1}",
                tools=[],
                config={},
                ins=["user_query"],
                outs=["response"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="input_to_agent", source="user_input_1", target="agent_1")
    ]
    
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Simple User Input Test",
        description="Testing data flow from user_input to agent",
        nodes=nodes,
        edges=edges,
        metadata={}
    )


async def run_test():
    """Run the simple test and trace data flow."""
    print("🧪 DAG EXECUTION FLOW TEST")
    print("=" * 50)
    
    # Create workflow
    workflow = create_simple_test_workflow()
    
    # Create DAG executor with execution metadata including user inputs
    execution_metadata_by_node = {
        "user_input_1": {
            "user_inputs": {
                "user_input_1": "What is the meaning of life?"
            }
        }
    }
    
    executor = create_dag_executor_from_spec(workflow)
    
    # Manually add execution metadata after creation
    # This simulates what the web interface should be doing
    for node_id, metadata in execution_metadata_by_node.items():
        if node_id in executor.nodes:
            dag_node = executor.nodes[node_id]
            # Update the task data with execution metadata
            task_data = dag_node.task_node_class.task
            if "execution_metadata" not in task_data:
                task_data["execution_metadata"] = {}
            task_data["execution_metadata"].update(metadata)
            print(f"✅ Added execution_metadata to node {node_id}")
            print(f"   Metadata: {metadata}")
    
    # Execute
    initial_state = WorkflowState(
        conversation_id="test_dag_flow",
        initial_text="",
        results={}
    )
    
    print("\n🚀 Starting DAG execution...")
    
    try:
        # Execute with timeout to detect hanging
        final_state = await asyncio.wait_for(
            executor.execute_dag(initial_state),
            timeout=30.0  # 30 second timeout
        )
        
        # Display results
        print("\n📊 EXECUTION COMPLETED:")
        print("-" * 50)
        
        for node_id, result in final_state.results.items():
            print(f"\n🔸 {node_id}:")
            if hasattr(result, 'result'):
                print(f"   Result: {result.result}")
            elif hasattr(result, 'agent_response') and hasattr(result.agent_response, 'result'):
                print(f"   Agent Response: {result.agent_response.result}")
            else:
                print(f"   Raw Result: {result}")
        
        print("\n" + "=" * 50)
        print("✅ Test Complete - NO HANGING!")
        
    except asyncio.TimeoutError:
        print("\n" + "=" * 50)
        print("❌ EXECUTION HUNG - Timeout after 30 seconds")
        print("\nDEBUG INFO:")
        print(f"Nodes in DAG: {list(executor.nodes.keys())}")
        print(f"Execution order: {executor.execution_order}")
        
        # Check what state we have so far
        if hasattr(executor, '_last_state') and executor._last_state:
            print(f"\nPartial results: {executor._last_state.results}")
        
        raise


if __name__ == "__main__":
    asyncio.run(run_test())