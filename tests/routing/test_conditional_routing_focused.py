#!/usr/bin/env python3
"""
Focused test for conditional routing - fast execution.
Tests only the critical path without multiple scenarios.
"""

import asyncio
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "iointel" / "src" / "utilities" / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec, EdgeData
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState


async def test_conditional_routing_single_case():
    """Test one specific routing case to verify the fix works."""
    print("🧪 Testing Conditional Routing - Single Case")
    print("=" * 60)
    
    # Create minimal workflow
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Quick Routing Test",
        description="Test route_index based conditional routing",
        nodes=[
            NodeSpec(
                id="decision",
                type="decision",
                label="Router",
                data=NodeData(
                    agent_instructions="""Call conditional_gate with:
data: {"value": 100}
router_config: {
  "conditions": [
    {"field": "value", "operator": ">", "value": 50, "route": "high"}
  ],
  "default_route": "low"
}""",
                    tools=["conditional_gate"],
                    outs=["decision"]
                )
            ),
            NodeSpec(
                id="high_path",
                type="agent",
                label="High Path",
                data=NodeData(
                    agent_instructions="High value path executed!",
                    ins=["decision"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="low_path",
                type="agent",
                label="Low Path",
                data=NodeData(
                    agent_instructions="Low value path executed!",
                    ins=["decision"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="to_high",
                source="decision",
                target="high_path",
                data=EdgeData(route_index=0, route_label="high")
            ),
            EdgeSpec(
                id="to_low",
                source="decision",
                target="low_path",
                data=EdgeData(route_index=-1, route_label="low")  # -1 for default
            )
        ]
    )
    
    # Execute
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow_spec.nodes,
        edges=workflow_spec.edges,
        objective="Test routing"
    )
    
    print("\n📊 Execution plan:")
    summary = executor.get_execution_summary()
    for i, batch in enumerate(summary["execution_order"]):
        print(f"  Batch {i}: {batch}")
    
    # Execute
    print("\n🚀 Executing...")
    initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    print("\n📊 Results:")
    executed_nodes = []
    skipped_nodes = []
    
    for node_id in ["decision", "high_path", "low_path"]:
        if node_id in final_state.results:
            result = final_state.results[node_id]
            # Check if it's a skip result
            if isinstance(result, dict) and result.get("status") == "skipped":
                skipped_nodes.append(node_id)
            else:
                executed_nodes.append(node_id)
    
    print(f"  ✅ Executed: {executed_nodes}")
    print(f"  ⏭️  Skipped: {skipped_nodes}")
    
    # Verify routing worked
    assert "decision" in executed_nodes, "Decision node should execute"
    assert "high_path" in executed_nodes, "High path should execute (value > 50)"
    assert "low_path" in skipped_nodes, "Low path should be skipped"
    
    # Check for routing decision in logs
    if "decision" in final_state.results:
        decision_result = final_state.results["decision"]
        print(f"\n🔍 Decision result type: {type(decision_result)}")
        
        # Extract routing info
        if hasattr(decision_result, 'agent_response'):
            agent_resp = decision_result.agent_response
            if hasattr(agent_resp, 'tool_usage_results') and agent_resp.tool_usage_results:
                for tool_result in agent_resp.tool_usage_results:
                    if hasattr(tool_result, 'tool_name') and tool_result.tool_name == 'conditional_gate':
                        gate_result = tool_result.tool_result
                        print(f"  🎯 Route index: {getattr(gate_result, 'route_index', 'N/A')}")
                        print(f"  🎯 Routed to: {getattr(gate_result, 'routed_to', 'N/A')}")
    
    print("\n✅ Conditional routing test PASSED!")
    return True


async def main():
    """Run focused conditional routing test."""
    try:
        success = await test_conditional_routing_single_case()
        if success:
            print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
            return 0
        else:
            print("\n❌ TEST FAILED!")
            return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)