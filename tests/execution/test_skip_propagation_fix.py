import os
#!/usr/bin/env python3
"""
Test Skip Propagation Through DAG Dependencies
=============================================
This tests that when decision gates block nodes, the skip status
propagates correctly to all downstream dependencies.

Case: decision_agent routes to neutral_confirmation
→ positive_agent & negative_agent are skipped
→ email_agent (depends on both) should also be skipped
"""
import asyncio
import sys
import uuid
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec, EdgeData, SLARequirements
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.datamodels import AgentParams
import os

# Import tools to register them

async def test_skip_propagation():
    """Test that skip status propagates through dependencies correctly."""
    
    print("🧪 TESTING SKIP PROPAGATION THROUGH DAG")
    print("=" * 50)
    
    # Create a workflow that mimics your sentiment analysis case
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Skip Propagation Test", 
        description="Test skip status propagation through dependencies",
        nodes=[
            # Prompt tool for neutral "do nothing" input (should skip all agents)
            NodeSpec(
                id="data_input", 
                type="tool",
                label="Data Input",
                data=NodeData(
                    tool_name="prompt_tool",
                    config={"message": "do nothing with TSLA, just hold"}
                )
            ),
            # Decision agent with conditional_gate
            NodeSpec(
                id="decision_agent", 
                type="agent",
                label="Decision Agent",
                data=NodeData(
                    agent_instructions="Analyze sentiment and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
                    tools=["conditional_gate"],
                    model="gpt-4o",
                    sla=SLARequirements(
                        enforce_usage=True,
                        required_tools=["conditional_gate"],
                        tool_usage_required=True
                    )
                )
            ),
            # Positive agent (should be skipped)
            NodeSpec(
                id="positive_agent", 
                type="agent",
                label="Positive Agent",
                data=NodeData(
                    agent_instructions="Handle positive sentiment confirmation",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Negative agent (should be skipped)
            NodeSpec(
                id="negative_agent", 
                type="agent",
                label="Negative Agent",
                data=NodeData(
                    agent_instructions="Handle negative sentiment confirmation",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Email agent (should be skipped because both dependencies are skipped)
            NodeSpec(
                id="email_agent", 
                type="agent",
                label="Email Agent",
                data=NodeData(
                    agent_instructions="Send email about sentiment analysis results",
                    tools=[],
                    model="gpt-4o"
                )
            )
        ],
        edges=[
            # Input flows to decision
            EdgeSpec(
                id="edge_1",
                source="data_input",
                target="decision_agent",
                data=EdgeData()
            ),
            # Decision routes to positive (will be skipped)
            EdgeSpec(
                id="edge_2",
                source="decision_agent",
                target="positive_agent",
                data=EdgeData(condition="routed_to == 'positive'")
            ),
            # Decision routes to negative (will be skipped)
            EdgeSpec(
                id="edge_3",
                source="decision_agent",
                target="negative_agent",
                data=EdgeData(condition="routed_to == 'negative'")
            ),
            # Both agents feed into email (should propagate skip)
            EdgeSpec(
                id="edge_4",
                source="positive_agent",
                target="email_agent",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge_5",
                source="negative_agent",
                target="email_agent",
                data=EdgeData()
            )
        ]
    )
    
    # Create agents (no agent needed for prompt_tool)
    agents = [
        AgentParams(
            name="decision_agent",
            instructions="Analyze sentiment and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
            tools=["conditional_gate"]
        ),
        AgentParams(
            name="positive_agent",
            instructions="Handle positive sentiment confirmation",
            tools=[]
        ),
        AgentParams(
            name="negative_agent",
            instructions="Handle negative sentiment confirmation", 
            tools=[]
        ),
        AgentParams(
            name="email_agent",
            instructions="Send email about sentiment analysis results",
            tools=[]
        )
    ]
    
    # Create DAG executor
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow.nodes,
        edges=workflow.edges,
        agents=agents,
        conversation_id="test_skip_propagation"
    )
    
    # Show execution plan
    print("📋 Execution Plan:")
    for i, batch in enumerate(executor.execution_order):
        print(f"  Batch {i}: {batch}")
    
    # Execute workflow
    initial_state = WorkflowState(
        initial_text="Test skip propagation",
        conversation_id="test_skip_propagation",
        results={}
    )
    
    try:
        print("\n🚀 Executing workflow...")
        final_state = await executor.execute_dag(initial_state)
        
        print("\n📊 EXECUTION RESULTS:")
        for node_id, result in final_state.results.items():
            if isinstance(result, dict) and "status" in result:
                status = result["status"]
                reason = result.get("reason", "")
                print(f"  {node_id}: {status} {f'({reason})' if reason else ''}")
            else:
                print(f"  {node_id}: executed successfully")
        
        # Check skip propagation
        print("\n🔍 SKIP PROPAGATION ANALYSIS:")
        
        # Decision agent should execute and route
        if "decision_agent" in final_state.results:
            decision_result = final_state.results["decision_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    routed_to = getattr(gate_result, 'routed_to', 'unknown')
                    print(f"  ✅ Decision routed to: {routed_to}")
        
        # Check if positive/negative agents are skipped
        positive_skipped = (final_state.results.get("positive_agent", {}).get("status") == "skipped")
        negative_skipped = (final_state.results.get("negative_agent", {}).get("status") == "skipped")
        email_skipped = (final_state.results.get("email_agent", {}).get("status") == "skipped")
        
        print(f"  ✅ positive_agent skipped: {positive_skipped}")
        print(f"  ✅ negative_agent skipped: {negative_skipped}")
        print(f"  🎯 email_agent skipped: {email_skipped}")
        
        if positive_skipped and negative_skipped and email_skipped:
            print("\n🎉 SKIP PROPAGATION WORKING CORRECTLY!")
            print("   Both dependencies skipped → email_agent correctly skipped")
        elif positive_skipped and negative_skipped and not email_skipped:
            print("\n❌ SKIP PROPAGATION BUG DETECTED!")
            print("   Both dependencies skipped but email_agent still executed")
            print("   This is the bug we need to fix!")
        else:
            print("\n⚠️ Unexpected result pattern:")
            print(f"   positive_skipped: {positive_skipped}")
            print(f"   negative_skipped: {negative_skipped}")
            print(f"   email_skipped: {email_skipped}")
        
        # Show execution statistics
        stats = executor.get_execution_statistics()
        print("\n📈 EXECUTION STATISTICS:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Executed: {stats['executed_nodes']}")
        print(f"  Skipped: {stats['skipped_nodes']}")
        print(f"  Efficiency: {stats['execution_efficiency']}")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_skip_propagation())