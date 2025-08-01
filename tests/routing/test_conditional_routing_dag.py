#!/usr/bin/env python3
"""
Test conditional routing with DAG executor.

This test demonstrates:
1. Decision nodes using conditional_gate tool
2. Proper route_index matching in DAG executor
3. Selective execution of downstream nodes based on routing
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
from iointel.src.utilities.dag_executor import DAGExecutor, create_dag_executor_from_spec
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.decorators import register_custom_task
from iointel.src.agent_methods.tools import conditional_gate  # Import to register the tool


async def test_conditional_routing_with_route_index():
    """Test conditional routing using route_index matching."""
    print("🧪 Testing Conditional Routing with route_index...")
    
    # Create workflow with decision node and conditional routing
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Conditional Routing Test",
        description="Test route_index based conditional routing",
        nodes=[
            NodeSpec(
                id="input_data",
                type="data_source",
                label="Input Data",
                data=NodeData(
                    source_name="prompt_tool",
                    config={"message": "Market sentiment: bullish, confidence: 0.85"},
                    outs=["market_data"]
                )
            ),
            NodeSpec(
                id="decision_node",
                type="decision",
                label="Market Decision",
                data=NodeData(
                    agent_instructions="""You are a market decision agent. Analyze the market data and use the conditional_gate tool to make a routing decision.

Use the conditional_gate tool with these conditions:
- If sentiment contains 'bullish' AND confidence > 0.7, route to 'buy' (route 0)
- If sentiment contains 'bearish' AND confidence > 0.7, route to 'sell' (route 1)  
- Otherwise route to 'hold' (route 2)

You MUST call the conditional_gate tool with proper router_config.""",
                    tools=["conditional_gate"],
                    ins=["market_data"],
                    outs=["routing_decision"]
                )
            ),
            NodeSpec(
                id="buy_action",
                type="agent",
                label="Buy Agent",
                data=NodeData(
                    agent_instructions="Execute BUY order. The market decision was bullish with high confidence.",
                    ins=["routing_decision"],
                    outs=["buy_result"]
                )
            ),
            NodeSpec(
                id="sell_action",
                type="agent",
                label="Sell Agent",
                data=NodeData(
                    agent_instructions="Execute SELL order. The market decision was bearish with high confidence.",
                    ins=["routing_decision"],
                    outs=["sell_result"]
                )
            ),
            NodeSpec(
                id="hold_action",
                type="agent",
                label="Hold Agent",
                data=NodeData(
                    agent_instructions="HOLD position. The market signals are unclear or confidence is low.",
                    ins=["routing_decision"],
                    outs=["hold_result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="input_to_decision", source="input_data", target="decision_node"),
            EdgeSpec(
                id="decision_to_buy",
                source="decision_node",
                target="buy_action",
                data=EdgeData(route_index=0, route_label="buy")
            ),
            EdgeSpec(
                id="decision_to_sell",
                source="decision_node",
                target="sell_action",
                data=EdgeData(route_index=1, route_label="sell")
            ),
            EdgeSpec(
                id="decision_to_hold",
                source="decision_node",
                target="hold_action",
                data=EdgeData(route_index=2, route_label="hold")
            )
        ]
    )
    
    # Create executor
    executor = create_dag_executor_from_spec(workflow_spec)
    
    # Verify execution plan
    summary = executor.get_execution_summary()
    print(f"  📊 Execution Summary: {summary}")
    
    assert summary["total_nodes"] == 5
    assert summary["total_batches"] == 3  # input → decision → (buy|sell|hold)
    
    # Execute DAG
    print("  🚀 Executing conditional routing workflow...")
    initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    # Check results
    print("\n  📊 Execution Results:")
    
    # Decision node should have executed
    assert "decision_node" in final_state.results
    decision_result = final_state.results["decision_node"]
    print(f"  ✅ Decision node executed")
    
    # Check which route was taken
    executed_nodes = [node_id for node_id in final_state.results if node_id not in ["input_data", "decision_node"]]
    skipped_nodes = [node_id for node_id in ["buy_action", "sell_action", "hold_action"] if node_id not in final_state.results]
    
    print(f"  🎯 Executed action nodes: {executed_nodes}")
    print(f"  ⏭️  Skipped action nodes: {skipped_nodes}")
    
    # Exactly one action node should have executed
    assert len(executed_nodes) == 1, f"Expected 1 action node to execute, got {len(executed_nodes)}"
    assert len(skipped_nodes) == 2, f"Expected 2 action nodes to be skipped, got {len(skipped_nodes)}"
    
    # Based on bullish sentiment with 0.85 confidence, buy_action should execute
    assert "buy_action" in executed_nodes, "Expected buy_action to execute for bullish sentiment"
    
    print("\n  ✅ Conditional routing works correctly!")
    print("  ✅ Only the selected route was executed!")


async def test_multiple_routing_scenarios():
    """Test different routing scenarios."""
    print("\n🧪 Testing Multiple Routing Scenarios...")
    
    scenarios = [
        {
            "name": "Bullish High Confidence",
            "data": "Market sentiment: bullish, confidence: 0.9",
            "expected_route": "buy_action"
        },
        {
            "name": "Bearish High Confidence", 
            "data": "Market sentiment: bearish, confidence: 0.8",
            "expected_route": "sell_action"
        },
        {
            "name": "Neutral Low Confidence",
            "data": "Market sentiment: neutral, confidence: 0.4",
            "expected_route": "hold_action"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  📋 Testing scenario: {scenario['name']}")
        
        # Create workflow with different input data
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title=f"Routing Test - {scenario['name']}",
            description="Test conditional routing",
            nodes=[
                NodeSpec(
                    id="input_data",
                    type="data_source",
                    label="Input Data",
                    data=NodeData(
                        source_name="prompt_tool",
                        config={"message": scenario['data']},
                        outs=["market_data"]
                    )
                ),
                NodeSpec(
                    id="decision_node",
                    type="decision",
                    label="Market Decision",
                    data=NodeData(
                        agent_instructions="""Analyze market data and route using conditional_gate tool.
Route to 'buy' (0) for bullish >0.7, 'sell' (1) for bearish >0.7, else 'hold' (2).

Extract the sentiment and confidence from the market data, then call conditional_gate with:
{
    "conditions": [
        {"field": "sentiment", "operator": "in", "value": ["bullish"], "route": "buy"},
        {"field": "sentiment", "operator": "in", "value": ["bearish"], "route": "sell"}
    ],
    "default_route": "hold"
}""",
                        tools=["conditional_gate"],
                        ins=["market_data"],
                        outs=["routing_decision"]
                    )
                ),
                NodeSpec(
                    id="buy_action",
                    type="agent",
                    label="Buy Agent",
                    data=NodeData(
                        agent_instructions="Execute BUY order",
                        ins=["routing_decision"],
                        outs=["buy_result"]
                    )
                ),
                NodeSpec(
                    id="sell_action",
                    type="agent",
                    label="Sell Agent",
                    data=NodeData(
                        agent_instructions="Execute SELL order",
                        ins=["routing_decision"],
                        outs=["sell_result"]
                    )
                ),
                NodeSpec(
                    id="hold_action",
                    type="agent",
                    label="Hold Agent",
                    data=NodeData(
                        agent_instructions="HOLD position",
                        ins=["routing_decision"],
                        outs=["hold_result"]
                    )
                )
            ],
            edges=[
                EdgeSpec(id="e1", source="input_data", target="decision_node"),
                EdgeSpec(id="e2", source="decision_node", target="buy_action", 
                        data=EdgeData(route_index=0)),
                EdgeSpec(id="e3", source="decision_node", target="sell_action",
                        data=EdgeData(route_index=1)),
                EdgeSpec(id="e4", source="decision_node", target="hold_action",
                        data=EdgeData(route_index=2))
            ]
        )
        
        # Execute
        executor = create_dag_executor_from_spec(workflow_spec)
        initial_state = WorkflowState(conversation_id=f"test_{scenario['name']}", initial_text="", results={})
        final_state = await executor.execute_dag(initial_state)
        
        # Verify correct routing
        executed_action = None
        for action in ["buy_action", "sell_action", "hold_action"]:
            if action in final_state.results:
                result = final_state.results[action]
                if not (isinstance(result, dict) and result.get("status") == "skipped"):
                    executed_action = action
                    break
        
        print(f"    Expected: {scenario['expected_route']}")
        print(f"    Executed: {executed_action}")
        assert executed_action == scenario['expected_route'], f"Expected {scenario['expected_route']}, got {executed_action}"
        print(f"    ✅ Routing correct!")
    
    print("\n  ✅ All routing scenarios passed!")


async def main():
    """Run all conditional routing tests."""
    print("🚀 Conditional Routing DAG Tests")
    print("=" * 60)
    
    try:
        await test_conditional_routing_with_route_index()
        await test_multiple_routing_scenarios()
        
        print("\n🎉 ALL CONDITIONAL ROUTING TESTS PASSED!")
        print("✅ route_index matching working")
        print("✅ Selective node execution based on routing")
        print("✅ Multiple routing scenarios handled correctly")
        print("✅ Decision nodes properly control workflow paths")
        
    except Exception as e:
        print(f"\n❌ CONDITIONAL ROUTING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())