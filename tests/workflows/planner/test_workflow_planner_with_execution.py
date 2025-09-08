import os
#!/usr/bin/env python3
"""
Test Workflow Planner with Execution
===================================
This tests the complete pipeline:
1. Natural language prompt → WorkflowPlanner → DAG generation
2. DAG → DAGExecutor → Execution with proper routing
3. Verification of execution modes, SLA, and skip propagation
"""
import asyncio
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

from iointel.src.utilities.workflow_helpers import plan_and_execute, analyze_workflow_spec
import os

async def test_complete_workflow_pipeline():
    """Test the complete workflow generation and execution pipeline."""
    
    print("🧪 TESTING COMPLETE WORKFLOW PIPELINE")
    print("=" * 60)
    
    # Test cases that verify our new features
    test_cases = [
        {
            "name": "Sentiment Routing with For-Each Email",
            "prompt": """Create a workflow that:
            1. Takes user input about a stock
            2. Analyzes sentiment to determine if it's positive or negative
            3. Routes to either a buy recommendation agent or sell recommendation agent
            4. Sends an email notification with the recommendation
            
            Important: The email should be sent regardless of which path is taken.
            The sentiment analyzer MUST use conditional_gate to route properly.""",
            "expected": {
                "decision_gates": 1,  # sentiment analyzer
                "for_each_nodes": ["email", "notification"],  # email agent should be for_each
                "sla_nodes": 1,  # sentiment analyzer should have SLA
                "conditional_edges": 2  # routing to buy/sell
            }
        },
        {
            "name": "Research Before Decision",
            "prompt": """Create a workflow where:
            1. A research agent searches for information about TSLA
            2. The same agent MUST use conditional_gate to route based on findings
            3. Route to either 'bullish analysis' or 'bearish analysis' agents
            4. Send a summary email with the analysis
            
            Ensure the research agent is required to use both search and conditional_gate tools.""",
            "expected": {
                "sla_nodes": 1,  # research agent with SLA
                "required_tools": ["search", "conditional_gate"],
                "final_tool": "conditional_gate",
                "for_each_nodes": ["summary", "email"]
            }
        },
        {
            "name": "Parallel Sources with Consolidation",
            "prompt": """Create a workflow that:
            1. Gets cryptocurrency prices from two different sources in parallel
            2. A comparison agent waits for BOTH price inputs
            3. Compares the prices and outputs the difference
            4. Sends an alert if the price difference is significant
            
            The comparison agent must wait for both inputs before running.""",
            "expected": {
                "parallel_nodes": 2,  # two price sources
                "consolidate_nodes": ["comparison", "compare"],  # comparison should consolidate
                "no_for_each": True  # no for_each nodes in this workflow
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"📋 Test {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Run the complete pipeline
        result = await plan_and_execute(
            prompt=test_case['prompt'],
            debug=False  # Set to True for detailed logs
        )
        
        if result["success"]:
            spec = result["workflow_spec"]
            stats = result["execution_stats"]
            final_state = result["execution_result"]
            
            print(f"\n✅ Workflow Generated: {spec.title}")
            print(f"📝 Description: {spec.description}")
            
            # Analyze the generated workflow
            analysis = analyze_workflow_spec(spec)
            
            print("\n🔍 Workflow Analysis:")
            print(f"  - Total nodes: {analysis['total_nodes']}")
            print(f"  - Node types: {analysis['node_types']}")
            print(f"  - For-each nodes: {analysis['execution_modes']['for_each']}")
            print(f"  - Consolidate nodes: {analysis['execution_modes']['consolidate']}")
            print(f"  - Decision gates: {analysis['decision_gates']}")
            print(f"  - SLA-enforced nodes: {len(analysis['sla_nodes'])}")
            
            # Show SLA details
            for sla_node in analysis['sla_nodes']:
                print(f"\n  📋 SLA for '{sla_node['node']}':")
                print(f"    - Required tools: {sla_node['required_tools']}")
                print(f"    - Final tool must be: {sla_node['final_tool_must_be']}")
                print(f"    - Min tool calls: {sla_node['min_tool_calls']}")
            
            # Show execution results
            print("\n🚀 Execution Results:")
            print(f"  - Executed nodes: {stats['executed_nodes']}")
            print(f"  - Skipped nodes: {stats['skipped_nodes']}")
            print(f"  - Efficiency: {stats['execution_efficiency']}")
            
            # Verify expectations
            print("\n✓ Verification:")
            expected = test_case.get("expected", {})
            
            # Check decision gates
            if "decision_gates" in expected:
                actual_gates = len(analysis['decision_gates'])
                expected_gates = expected["decision_gates"]
                status = "✅" if actual_gates >= expected_gates else "❌"
                print(f"  {status} Decision gates: {actual_gates} (expected >= {expected_gates})")
            
            # Check for-each nodes
            if "for_each_nodes" in expected:
                for_each_found = any(
                    keyword in node.lower() 
                    for node in analysis['execution_modes']['for_each']
                    for keyword in expected["for_each_nodes"]
                )
                status = "✅" if for_each_found else "❌"
                print(f"  {status} For-each email node: {for_each_found}")
            
            # Check SLA nodes
            if "sla_nodes" in expected:
                actual_sla = len(analysis['sla_nodes'])
                expected_sla = expected["sla_nodes"]
                status = "✅" if actual_sla >= expected_sla else "❌"
                print(f"  {status} SLA nodes: {actual_sla} (expected >= {expected_sla})")
            
            # Show some execution details
            if final_state and final_state.results:
                print("\n📊 Sample Execution Output:")
                for node_id, node_result in list(final_state.results.items())[:3]:
                    if isinstance(node_result, dict):
                        status = node_result.get("status", "completed")
                        print(f"  - {node_id}: {status}")
                        if "result" in node_result and isinstance(node_result["result"], str):
                            preview = node_result["result"][:80] + "..." if len(node_result["result"]) > 80 else node_result["result"]
                            print(f"    → {preview}")
            
        else:
            print(f"\n❌ Pipeline Failed: {result.get('error', 'Unknown error')}")
            if result.get('workflow_spec'):
                print("   (Workflow was generated but execution failed)")
    
    print(f"\n{'='*60}")
    print("🏁 All tests completed!")


async def test_specific_routing_scenario():
    """Test a specific routing scenario in detail."""
    
    print("\n🎯 TESTING SPECIFIC ROUTING SCENARIO")
    print("=" * 60)
    
    prompt = """
    Create a workflow for stock sentiment analysis:
    1. User provides stock ticker and sentiment text
    2. A sentiment analyzer agent uses conditional_gate to route to 'buy', 'sell', or 'hold'
    3. Buy path: Create detailed buy recommendation
    4. Sell path: Create detailed sell recommendation  
    5. Hold path: Create holding strategy
    6. Email agent sends the final recommendation
    
    Requirements:
    - Sentiment analyzer MUST use conditional_gate (enforce with SLA)
    - Email agent should run regardless of which path is taken (use for_each mode)
    - Include proper models and execution modes for all nodes
    """
    
    result = await plan_and_execute(prompt, debug=True)
    
    if result["success"]:
        print("\n✅ Detailed routing test passed!")
        
        # Check specific routing behavior
        result["execution_result"]
        stats = result["execution_stats"]
        
        print("\n📊 Routing Execution Summary:")
        print("  - Total paths possible: 3 (buy, sell, hold)")
        print(f"  - Nodes executed: {stats['executed_nodes']}")
        print(f"  - Nodes skipped: {stats['skipped_nodes']}")
        
        # Show which path was taken
        if stats['skipped_node_ids']:
            print(f"  - Skipped nodes: {stats['skipped_node_ids']}")
            print("  → This indicates proper routing (only one path executed)")
    
    else:
        print(f"\n❌ Routing test failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(test_complete_workflow_pipeline())
    # Uncomment to run the specific routing test
    # asyncio.run(test_specific_routing_scenario())