#!/usr/bin/env python3
"""
Test the diamond workflow to ensure tool resolution works.
"""

import asyncio
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "iointel" / "src" / "utilities" / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env


async def test_diamond_workflow():
    """Test a simple diamond workflow with math tools."""
    print("🧪 Testing Diamond Workflow")
    print("=" * 60)
    
    # Load tools from environment
    print("\n📦 Loading tools...")
    available_tools = load_tools_from_env(env_path)
    print(f"  Available tools: {len(available_tools)}")
    
    # Check if calculator tools are available
    calculator_tools = [t for t in available_tools if t.startswith("calculator_")]
    print(f"  Calculator tools: {calculator_tools}")
    
    # Simplified workflow - just test tool resolution
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Simple Diamond Test",
        description="Test tool resolution in diamond pattern",
        nodes=[
            NodeSpec(
                id="start",
                type="agent",
                label="Start",
                data=NodeData(
                    agent_instructions="Generate the number 50",
                    tools=[],  # No tools needed
                    outs=["number"]
                )
            ),
            NodeSpec(
                id="multiply",
                type="agent", 
                label="Multiply",
                data=NodeData(
                    agent_instructions="Multiply the input number by 2 using the calculator_multiply tool",
                    tools=["calculator_multiply"],
                    ins=["number"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="add",
                type="agent",
                label="Add",
                data=NodeData(
                    agent_instructions="Add 10 to the input number using the calculator_add tool",
                    tools=["calculator_add"],
                    ins=["number"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="combine",
                type="agent",
                label="Combine",
                data=NodeData(
                    agent_instructions="Add the two results together using calculator_add",
                    tools=["calculator_add"],
                    ins=["multiply_result", "add_result"],
                    outs=["final"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="e1", source="start", target="multiply"),
            EdgeSpec(id="e2", source="start", target="add"),
            EdgeSpec(id="e3", source="multiply", target="combine"),
            EdgeSpec(id="e4", source="add", target="combine")
        ]
    )
    
    # Execute with typed execution
    executor = DAGExecutor(use_typed_execution=True)
    executor.build_execution_graph(
        nodes=workflow_spec.nodes,
        edges=workflow_spec.edges,
        objective="Test diamond workflow"
    )
    
    print("\n📊 Execution plan:")
    summary = executor.get_execution_summary()
    for i, batch in enumerate(summary["execution_order"]):
        print(f"  Batch {i}: {batch}")
    
    try:
        print("\n🚀 Executing...")
        initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
        final_state = await executor.execute_dag(initial_state)
        
        print("\n✅ Execution completed successfully!")
        print(f"  Executed nodes: {list(final_state.results.keys())}")
        
        # Check if tools were used
        for node_id, result in final_state.results.items():
            if hasattr(result, 'agent_response') and result.agent_response:
                if hasattr(result.agent_response, 'tool_usage_results'):
                    tools_used = [t.tool_name for t in result.agent_response.tool_usage_results]
                    if tools_used:
                        print(f"  {node_id} used tools: {tools_used}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    success = await test_diamond_workflow()
    if success:
        print("\n🎉 DIAMOND WORKFLOW TEST PASSED!")
    else:
        print("\n❌ DIAMOND WORKFLOW TEST FAILED!")
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))