import os
#!/usr/bin/env python3
"""
Test Negative Sentiment Routing Through DAG
==========================================
This tests that when decision_agent routes to 'negative', only the
negative_agent executes and email_agent receives one response.

Case: decision_agent routes to negative
→ negative_agent executes 
→ positive_agent is skipped
→ email_agent executes (depends on both but negative provides input)
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

# Import tools to register them
from iointel.src.agent_methods.tools.user_input import prompt_tool
from iointel.src.agent_methods.tools import conditional_gate
import os

async def test_negative_routing():
    """Test that negative routing works correctly."""
    
    print("🧪 TESTING NEGATIVE SENTIMENT ROUTING")
    print("=" * 50)
    
    # Create a workflow that should route to negative
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Negative Routing Test", 
        description="Test negative sentiment routing",
        nodes=[
            # Prompt tool with negative sentiment
            NodeSpec(
                id="data_input", 
                type="tool",
                label="Data Input",
                data=NodeData(
                    tool_name="prompt_tool",
                    config={"message": "sell all TSLA shares immediately, avoid this stock!"}
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
                    agent_instructions="Handle positive sentiment confirmation for buying decision",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Negative agent (should execute)
            NodeSpec(
                id="negative_agent", 
                type="agent",
                label="Negative Agent",
                data=NodeData(
                    agent_instructions="Handle negative sentiment confirmation for selling decision",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Email agent (should execute with one input from negative_agent using for_each mode)
            NodeSpec(
                id="email_agent", 
                type="agent",
                label="Email Agent",
                data=NodeData(
                    agent_instructions="Send email about sentiment analysis results based on available inputs",
                    tools=[],
                    model="gpt-4o",
                    execution_mode="for_each"  # Execute for each completed dependency
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
            # Decision routes to positive
            EdgeSpec(
                id="edge_2",
                source="decision_agent",
                target="positive_agent",
                data=EdgeData(condition="routed_to == 'positive'")
            ),
            # Decision routes to negative
            EdgeSpec(
                id="edge_3",
                source="decision_agent",
                target="negative_agent",
                data=EdgeData(condition="routed_to == 'negative'")
            ),
            # Both agents feed into email
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
    
    # Create agents
    agents = [
        AgentParams(
            name="decision_agent",
            instructions="Analyze sentiment and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
            tools=["conditional_gate"]
        ),
        AgentParams(
            name="positive_agent",
            instructions="Handle positive sentiment confirmation for buying decision",
            tools=[]
        ),
        AgentParams(
            name="negative_agent",
            instructions="Handle negative sentiment confirmation for selling decision", 
            tools=[]
        ),
        AgentParams(
            name="email_agent",
            instructions="Send email about sentiment analysis results based on available inputs",
            tools=[]
        )
    ]
    
    # Create DAG executor
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow.nodes,
        edges=workflow.edges,
        agents=agents,
        conversation_id="test_negative_routing"
    )
    
    # Show execution plan
    print("📋 Execution Plan:")
    for i, batch in enumerate(executor.execution_order):
        print(f"  Batch {i}: {batch}")
    
    # Execute workflow
    initial_state = WorkflowState(
        initial_text="Test negative routing",
        conversation_id="test_negative_routing",
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
        
        # Check routing behavior
        print("\n🔍 NEGATIVE ROUTING ANALYSIS:")
        
        # Decision agent should route to negative
        if "decision_agent" in final_state.results:
            decision_result = final_state.results["decision_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    routed_to = getattr(gate_result, 'routed_to', 'unknown')
                    print(f"  ✅ Decision routed to: {routed_to}")
        
        # Check execution status
        positive_skipped = (final_state.results.get("positive_agent", {}).get("status") == "skipped")
        negative_executed = "negative_agent" in final_state.results and final_state.results["negative_agent"].get("status") != "skipped"
        email_executed = "email_agent" in final_state.results and final_state.results["email_agent"].get("status") != "skipped"
        
        print(f"  ✅ positive_agent skipped: {positive_skipped}")
        print(f"  ✅ negative_agent executed: {negative_executed}")
        print(f"  🎯 email_agent executed: {email_executed}")
        
        if positive_skipped and negative_executed and email_executed:
            print("\n🎉 NEGATIVE ROUTING WORKING CORRECTLY!")
            print("   negative_agent executed → positive_agent skipped → email_agent got one input")
        else:
            print("\n❌ UNEXPECTED ROUTING BEHAVIOR:")
            print(f"   positive_skipped: {positive_skipped}")
            print(f"   negative_executed: {negative_executed}")
            print(f"   email_executed: {email_executed}")
        
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
    asyncio.run(test_negative_routing())