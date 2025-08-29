#!/usr/bin/env python3
"""Test decision node prompt injection with SLA enforcement."""

import asyncio
import uuid
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, SLARequirements
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agents import Agent
from iointel.src.utilities.io_logger import system_logger

async def test_decision_prompt_injection():
    """Test that decision nodes get proper prompt injection."""
    
    # Create a simple workflow with a decision node
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        reasoning="Testing decision node prompt injection",
        title="Test Decision Prompt Injection",
        description="Test workflow for decision prompt injection",
        nodes=[
            NodeSpec(
                id="input_1",
                type="data_source",
                label="Get user query",
                data=NodeData(
                    source_name="user_input",
                    config={
                        "message": "What would you like to analyze?",
                        "default_value": "Analyze AAPL stock"
                    }
                )
            ),
            NodeSpec(
                id="decision_1",
                type="decision",
                label="Route based on query",
                data=NodeData(
                    agent_instructions="Analyze the user query and determine whether it's asking for stock analysis, weather, or something else.",
                    tools=["conditional_gate"],
                    ins=["input_1"]
                ),
                sla=SLARequirements(
                    tool_usage_required=True,
                    required_tools=["conditional_gate"],
                    final_tool_must_be="conditional_gate",
                    min_tool_calls=1,
                    enforce_usage=True
                )
            ),
            NodeSpec(
                id="stock_agent",
                type="agent",
                label="Stock Analysis",
                data=NodeData(
                    agent_instructions="Perform stock analysis for {input_1}",
                    tools=["get_stock_data"],
                    ins=["decision_1"]
                )
            ),
            NodeSpec(
                id="weather_agent",
                type="agent",
                label="Weather Analysis",
                data=NodeData(
                    agent_instructions="Get weather information for {input_1}",
                    tools=["get_weather"],
                    ins=["decision_1"]
                )
            ),
            NodeSpec(
                id="general_agent",
                type="agent",
                label="General Response",
                data=NodeData(
                    agent_instructions="Provide a general response to {input_1}",
                    ins=["decision_1"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="e1",
                source="input_1",
                target="decision_1"
            ),
            EdgeSpec(
                id="e2",
                source="decision_1",
                target="stock_agent",
                data={"route_index": 0, "route_label": "stock"}
            ),
            EdgeSpec(
                id="e3",
                source="decision_1",
                target="weather_agent",
                data={"route_index": 1, "route_label": "weather"}
            ),
            EdgeSpec(
                id="e4",
                source="decision_1",
                target="general_agent",
                data={"route_index": 2, "route_label": "other"}
            )
        ]
    )
    
    # Create initial state with user inputs
    initial_state = WorkflowState(
        initial_text="Test decision routing",
        conversation_id="test-decision-123",
        user_inputs={"input_1": "What's the weather like in San Francisco?"}
    )
    
    # We'll check the logs to see what instructions the agent gets
    # Since we can't easily mock the agent creation, we'll just run and observe
    mock_agents = []  # Let the typed executor create agents
    
    # Execute with DAG executor
    executor = DAGExecutor(
        workflow_spec=workflow_spec,
        use_typed_execution=True  # Use new typed execution
    )
    
    try:
        final_state = await executor.execute_dag(
            initial_state=initial_state,
            agents=mock_agents
        )
        
        print("\n=== Test Results ===")
        print(f"Execution completed successfully")
        print(f"Final results: {list(final_state.results.keys())}")
        
        # Check if decision node was executed
        if "decision_1" in final_state.results:
            result = final_state.results["decision_1"]
            print(f"\nDecision node result: {result}")
            
            # The key test: did the decision agent get enhanced instructions?
            # We'll need to check the logs to see the actual instructions
            print("\nCheck the logs above to see if the decision agent received enhanced instructions with SLA requirements.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_decision_prompt_injection())