"""Test the new typed execution system."""

import asyncio
import sys
sys.path.append('.')

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.io_logger import get_component_logger

logger = get_component_logger("TEST_TYPED_EXEC")


async def test_typed_execution():
    """Test that typed execution works for a simple workflow."""
    
    logger.info("🧪 Testing typed execution system")
    
    # Create a simple workflow: user_input -> agent
    import uuid
    workflow_spec = WorkflowSpec(
        id=str(uuid.uuid4()),
        rev=0,
        title="Test Typed Execution",
        name="Test Typed Execution",
        description="Test the new typed execution system",
        nodes=[
            NodeSpec(
                id="user_input_1",
                type="data_source",
                label="User Input",
                data=NodeData(
                    source_name="user_input",
                    config={
                        "message": "What would you like to talk about?",
                        "default_value": "Hello, I love bubble gum!"
                    },
                    ins=[],
                    outs=["user_message"]
                )
            ),
            NodeSpec(
                id="chat_agent_1",
                type="agent",
                label="Chat Agent",
                data=NodeData(
                    agent_instructions="You are a friendly assistant. Respond to the user's message.",
                    tools=[],
                    model="gpt-4o",
                    ins=["user_message"],
                    outs=["response"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="edge_1",
                source="user_input_1",
                target="chat_agent_1",
                data=EdgeData()
            )
        ]
    )
    
    # Test both execution modes
    for use_typed in [False, True]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with typed execution: {use_typed}")
        logger.info(f"{'='*60}")
        
        # Create DAG executor
        executor = DAGExecutor(use_typed_execution=use_typed)
        
        # Build execution graph
        executor.build_execution_graph(
            nodes=workflow_spec.nodes,
            edges=workflow_spec.edges,
            objective="Test workflow",
            agents=[],  # Will be auto-created from node data
            conversation_id=f"test-typed-{use_typed}"
        )
        
        # Create initial state
        initial_state = WorkflowState(
            results={},
            conversation_id=f"test-typed-{use_typed}"
        )
        
        # Execute
        try:
            final_state = await executor.execute_dag(initial_state)
            
            logger.info(f"\n✅ Execution completed successfully!")
            logger.info(f"Results: {list(final_state.results.keys())}")
            
            # Check user input result
            if "user_input_1_result" in final_state.results:
                user_input_result = final_state.results["user_input_1_result"]
                logger.info(f"\nUser input result: {user_input_result}")
            
            # Check agent result
            if "chat_agent_1_result" in final_state.results:
                agent_result = final_state.results["chat_agent_1_result"]
                logger.info(f"\nAgent result type: {type(agent_result)}")
                if hasattr(agent_result, 'agent_response') and agent_result.agent_response:
                    response_text = agent_result.agent_response.result
                    logger.info(f"Agent response: {response_text}")
                    # Check if agent mentioned bubble gum
                    if "bubble" in response_text.lower() or "gum" in response_text.lower():
                        logger.info("✅ Agent correctly responded about bubble gum!")
                    else:
                        logger.warning("⚠️  Agent did not mention bubble gum in response")
                else:
                    logger.info(f"Agent result: {str(agent_result)[:200]}...")
            
        except Exception as e:
            logger.error(f"\n❌ Execution failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n🏁 Test complete!")


if __name__ == "__main__":
    asyncio.run(test_typed_execution())