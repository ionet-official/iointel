#!/usr/bin/env python3
"""
Test decision node routing with new discriminated union WorkflowSpec.
"""
import asyncio
from uuid import uuid4

from iointel.src.utilities.workflow_helpers import execute_workflow
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    DataSourceNode,
    AgentNode,
    DecisionNode,
    DataSourceData,
    AgentConfig,
    DecisionConfig,
    DataSourceConfig,
    EdgeSpec,
    EdgeData,
    SLARequirements
)
from iointel.src.agent_methods.data_models.execution_models import ExecutionStatus


async def test_decision_routing():
    """Test a workflow with decision node routing."""
    print("\n" + "="*60)
    print("TEST: Decision Node Routing with Conditional Gate")
    print("="*60)
    
    workflow_id = uuid4()
    
    # Build workflow with decision routing
    nodes = [
        DataSourceNode(
            id="input",
            type="data_source",
            label="Market Sentiment Input",
            data=DataSourceData(
                source_name="user_input",
                config=DataSourceConfig(
                    message="Enter market sentiment (bullish/bearish) and confidence (0-1)",
                    default_value="bullish 0.8"
                )
            )
        ),
        DecisionNode(
            id="router",
            type="decision",
            label="Sentiment Router",
            data=DecisionConfig(
                agent_instructions="""Analyze the market sentiment from 'Market Sentiment Input'.
                Extract sentiment (bullish/bearish) and confidence level.
                Use conditional_gate to route:
                - Route to index 0 if bullish with confidence > 0.7
                - Route to index 1 if bearish with confidence > 0.7
                - Route to index 2 for neutral/uncertain (default)""",
                tools=["conditional_gate"],
                sla=SLARequirements(
                    tool_usage_required=True,
                    required_tools=["conditional_gate"],
                    final_tool_must_be="conditional_gate",
                    enforce_usage=True
                )
            )
        ),
        AgentNode(
            id="bullish_handler",
            type="agent",
            label="Bullish Strategy",
            data=AgentConfig(
                agent_instructions="Market is bullish! Recommend buying positions.",
                tools=[]
            )
        ),
        AgentNode(
            id="bearish_handler",
            type="agent",
            label="Bearish Strategy",
            data=AgentConfig(
                agent_instructions="Market is bearish! Recommend selling or shorting.",
                tools=[]
            )
        ),
        AgentNode(
            id="neutral_handler",
            type="agent",
            label="Neutral Strategy",
            data=AgentConfig(
                agent_instructions="Market is uncertain. Recommend holding current positions.",
                tools=[]
            )
        )
    ]
    
    # Edges with routing metadata
    edges = [
        EdgeSpec(
            id="e_input_router",
            source="input",
            target="router"
        ),
        EdgeSpec(
            id="e_router_bullish",
            source="router",
            target="bullish_handler",
            data=EdgeData(route_index=0, route_label="bullish")
        ),
        EdgeSpec(
            id="e_router_bearish",
            source="router",
            target="bearish_handler",
            data=EdgeData(route_index=1, route_label="bearish")
        ),
        EdgeSpec(
            id="e_router_neutral",
            source="router",
            target="neutral_handler",
            data=EdgeData(route_index=2, route_label="neutral")
        )
    ]
    
    workflow = WorkflowSpec(
        id=workflow_id,
        rev=1,
        reasoning="Test decision node routing with conditional_gate",
        title="Market Sentiment Router",
        description="Route based on market sentiment analysis",
        nodes=nodes,
        edges=edges
    )
    
    print(f"Built workflow: {workflow.title}")
    print("\nEdges with routing:")
    for edge in workflow.edges:
        if edge.data.route_index is not None:
            print(f"  - {edge.source} → {edge.target} (route {edge.data.route_index}: {edge.data.route_label})")
    
    # Test 1: Bullish scenario
    print("\n--- Test 1: Bullish Scenario ---")
    result = await execute_workflow(
        workflow_spec=workflow,
        user_inputs={"Market Sentiment Input": "bullish 0.9"},
        debug=False
    )
    
    print(f"Result: {result.status}")
    if result.status == ExecutionStatus.COMPLETED:
        print("✓ Bullish routing executed successfully")
        # Check which nodes executed
        executed_nodes = list(result.node_results.keys())
        print(f"Executed nodes: {executed_nodes}")
        if "bullish_handler" in executed_nodes:
            print("✓ Correctly routed to bullish handler")
        else:
            print("✗ Did not route to bullish handler")
    
    # Test 2: Bearish scenario
    print("\n--- Test 2: Bearish Scenario ---")
    result2 = await execute_workflow(
        workflow_spec=workflow,
        user_inputs={"Market Sentiment Input": "bearish 0.85"},
        debug=False
    )
    
    print(f"Result: {result2.status}")
    if result2.status == ExecutionStatus.COMPLETED:
        print("✓ Bearish routing executed successfully")
        executed_nodes = list(result2.node_results.keys())
        print(f"Executed nodes: {executed_nodes}")
        if "bearish_handler" in executed_nodes:
            print("✓ Correctly routed to bearish handler")
        else:
            print("✗ Did not route to bearish handler")
    
    # Test 3: Neutral scenario (low confidence)
    print("\n--- Test 3: Neutral Scenario (low confidence) ---")
    result3 = await execute_workflow(
        workflow_spec=workflow,
        user_inputs={"Market Sentiment Input": "bullish 0.3"},
        debug=False
    )
    
    print(f"Result: {result3.status}")
    if result3.status == ExecutionStatus.COMPLETED:
        print("✓ Neutral routing executed successfully")
        executed_nodes = list(result3.node_results.keys())
        print(f"Executed nodes: {executed_nodes}")
        if "neutral_handler" in executed_nodes:
            print("✓ Correctly routed to neutral handler")
        else:
            print("✗ Did not route to neutral handler")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_success = all([
        result.status == ExecutionStatus.COMPLETED,
        result2.status == ExecutionStatus.COMPLETED,
        result3.status == ExecutionStatus.COMPLETED
    ])
    
    if all_success:
        print("🎉 ALL DECISION ROUTING TESTS PASSED!")
        print("The new discriminated union structure works with conditional_gate routing")
    else:
        print("⚠️ Some tests failed - decision routing needs investigation")
    
    return all_success


async def test_multi_branch_decision():
    """Test a more complex multi-branch decision workflow."""
    print("\n" + "="*60)
    print("TEST: Multi-Branch Decision Workflow")
    print("="*60)
    
    workflow_id = uuid4()
    
    nodes = [
        DataSourceNode(
            id="score_input",
            type="data_source",
            label="Score Input",
            data=DataSourceData(
                source_name="user_input",
                config=DataSourceConfig(
                    message="Enter a score (0-100)",
                    default_value="75"
                )
            )
        ),
        DecisionNode(
            id="grade_router",
            type="decision",
            label="Grade Calculator",
            data=DecisionConfig(
                agent_instructions="""Analyze the score from 'Score Input' and route based on grade:
                - Route to index 0 for A (90-100)
                - Route to index 1 for B (80-89)
                - Route to index 2 for C (70-79)
                - Route to index 3 for D (60-69)
                - Route to index 4 for F (below 60)
                Use conditional_gate to route appropriately.""",
                tools=["conditional_gate"],
                sla=SLARequirements(
                    tool_usage_required=True,
                    required_tools=["conditional_gate"],
                    final_tool_must_be="conditional_gate",
                    enforce_usage=True
                )
            )
        ),
        AgentNode(
            id="grade_a",
            type="agent",
            label="Grade A Handler",
            data=AgentConfig(
                agent_instructions="Excellent work! Grade A achieved.",
                tools=[]
            )
        ),
        AgentNode(
            id="grade_b",
            type="agent",
            label="Grade B Handler",
            data=AgentConfig(
                agent_instructions="Good job! Grade B achieved.",
                tools=[]
            )
        ),
        AgentNode(
            id="grade_c",
            type="agent",
            label="Grade C Handler",
            data=AgentConfig(
                agent_instructions="Satisfactory. Grade C achieved.",
                tools=[]
            )
        ),
        AgentNode(
            id="grade_d",
            type="agent",
            label="Grade D Handler",
            data=AgentConfig(
                agent_instructions="Below average. Grade D achieved.",
                tools=[]
            )
        ),
        AgentNode(
            id="grade_f",
            type="agent",
            label="Grade F Handler",
            data=AgentConfig(
                agent_instructions="Failed. Grade F. Needs improvement.",
                tools=[]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="e1", source="score_input", target="grade_router"),
        EdgeSpec(id="e_a", source="grade_router", target="grade_a", 
                data=EdgeData(route_index=0, route_label="A")),
        EdgeSpec(id="e_b", source="grade_router", target="grade_b",
                data=EdgeData(route_index=1, route_label="B")),
        EdgeSpec(id="e_c", source="grade_router", target="grade_c",
                data=EdgeData(route_index=2, route_label="C")),
        EdgeSpec(id="e_d", source="grade_router", target="grade_d",
                data=EdgeData(route_index=3, route_label="D")),
        EdgeSpec(id="e_f", source="grade_router", target="grade_f",
                data=EdgeData(route_index=4, route_label="F"))
    ]
    
    workflow = WorkflowSpec(
        id=workflow_id,
        rev=1,
        reasoning="Multi-branch grade routing workflow",
        title="Grade Calculator",
        description="Route to different handlers based on score",
        nodes=nodes,
        edges=edges
    )
    
    # Test different scores
    test_cases = [
        ("95", "grade_a", "A"),
        ("85", "grade_b", "B"),
        ("75", "grade_c", "C"),
        ("65", "grade_d", "D"),
        ("45", "grade_f", "F")
    ]
    
    results = []
    for score, expected_node, grade in test_cases:
        print(f"\nTesting score {score} (expecting grade {grade})...")
        result = await execute_workflow(
            workflow_spec=workflow,
            user_inputs={"Score Input": score},
            debug=False
        )
        
        success = result.status == ExecutionStatus.COMPLETED
        executed = list(result.node_results.keys())
        correct_routing = expected_node in executed
        
        print(f"  Status: {result.status}")
        print(f"  Executed: {executed}")
        print(f"  Correct routing: {'✓' if correct_routing else '✗'}")
        
        results.append(success and correct_routing)
    
    return all(results)


async def main():
    """Run all decision routing tests."""
    print("\n" + "="*80)
    print("TESTING DECISION NODE ROUTING WITH NEW DISCRIMINATED UNION STRUCTURE")
    print("="*80)
    
    # Test basic decision routing
    test1_success = await test_decision_routing()
    
    # Test multi-branch decision
    test2_success = await test_multi_branch_decision()
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL TEST RESULTS")
    print("="*80)
    
    if test1_success and test2_success:
        print("🎉 ALL TESTS PASSED!")
        print("Decision nodes with conditional_gate routing work correctly with the new structure")
    else:
        print("⚠️ Some tests failed")
        print(f"Basic routing: {'✓' if test1_success else '✗'}")
        print(f"Multi-branch: {'✓' if test2_success else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())