#!/usr/bin/env python3
"""
Test multiple routing scenarios to ensure all cases work.
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


async def test_routing_scenario(scenario_name, test_value, expected_route):
    """Test a specific routing scenario."""
    print(f"\n📋 Testing: {scenario_name}")
    print("-" * 40)
    
    # Create workflow
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title=f"Test: {scenario_name}",
        description="Test routing",
        nodes=[
            NodeSpec(
                id="router",
                type="decision",
                label="Router",
                data=NodeData(
                    agent_instructions=f"""Call conditional_gate with:
data: {{"value": {test_value}}}
router_config: {{
  "conditions": [
    {{"field": "value", "operator": ">", "value": 75, "route": "high"}},
    {{"field": "value", "operator": ">", "value": 25, "route": "medium"}}
  ],
  "default_route": "low"
}}""",
                    tools=["conditional_gate"],
                    outs=["decision"]
                )
            ),
            NodeSpec(
                id="high",
                type="agent",
                label="High",
                data=NodeData(
                    agent_instructions="High route!",
                    ins=["decision"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="medium",
                type="agent",
                label="Medium",
                data=NodeData(
                    agent_instructions="Medium route!",
                    ins=["decision"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="low",
                type="agent",
                label="Low",
                data=NodeData(
                    agent_instructions="Low route!",
                    ins=["decision"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="to_high", source="router", target="high",
                    data=EdgeData(route_index=0, route_label="high")),
            EdgeSpec(id="to_medium", source="router", target="medium",
                    data=EdgeData(route_index=1, route_label="medium")),
            EdgeSpec(id="to_low", source="router", target="low",
                    data=EdgeData(route_index=-1, route_label="low"))
        ]
    )
    
    # Execute
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow_spec.nodes,
        edges=workflow_spec.edges,
        objective="Test routing"
    )
    
    initial_state = WorkflowState(conversation_id=f"test_{scenario_name}", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    # Check results
    executed = []
    skipped = []
    
    for node_id in ["high", "medium", "low"]:
        if node_id in final_state.results:
            result = final_state.results[node_id]
            if isinstance(result, dict) and result.get("status") == "skipped":
                skipped.append(node_id)
                print(f"  🚫 Route {node_id} was skipped")
            else:
                executed.append(node_id)
                print(f"  ✅ Route {node_id} was executed")
        else:
            skipped.append(node_id)
            print(f"  🚫 Route {node_id} was skipped (not in results)")
            
    print(f"  Value: {test_value}")
    print(f"  Expected: {expected_route}")
    print(f"  Executed: {executed}")
    print(f"  Skipped: {skipped}")
    
    # Verify
    assert expected_route in executed, f"Expected {expected_route} to execute"
    assert len(executed) == 1, f"Only one route should execute, got {len(executed)}"
    assert len(skipped) == 2, f"Two routes should be skipped, got {len(skipped)}"
    
    print("  ✅ PASSED!")
    return True


async def main():
    """Run all routing scenario tests."""
    print("🧪 Testing Multiple Routing Scenarios")
    print("=" * 60)
    
    scenarios = [
        ("High Value", 100, "high"),      # > 75
        ("Medium Value", 50, "medium"),   # > 25 but <= 75
        ("Low Value", 10, "low"),         # <= 25 (default)
        ("Boundary 75", 75, "medium"),    # Exactly 75 (not > 75)
        ("Boundary 25", 25, "low"),       # Exactly 25 (not > 25)
    ]
    
    try:
        for name, value, expected in scenarios:
            await test_routing_scenario(name, value, expected)
        
        print("\n🎉 ALL ROUTING SCENARIOS PASSED!")
        print("✅ Route index matching works correctly")
        print("✅ Only selected routes execute")
        print("✅ Boundary conditions handled properly")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)