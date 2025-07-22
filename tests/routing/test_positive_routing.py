#!/usr/bin/env python3
"""
Test Positive Sentiment Routing Through DAG
==========================================
This tests that when decision_agent routes to 'positive', only the
positive_agent executes and email_agent receives one response.

Case: decision_agent routes to positive
→ positive_agent executes 
→ negative_agent is skipped
→ email_agent executes (depends on both but positive provides input)

Can run in two modes:
1. Manual DAG construction (default)
2. Workflow planner generation (--use-planner)
"""
import asyncio
import sys
import uuid
sys.path.append('/Users/alexandermorisse/Documents/GitHub/iointel')

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec, EdgeData, SLARequirements
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.datamodels import AgentParams

# Import tools to register them

async def test_positive_routing():
    """Test that positive routing works correctly."""
    
    print("🧪 TESTING POSITIVE SENTIMENT ROUTING")
    print("=" * 50)
    
    # Create a workflow that should route to positive
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Positive Routing Test", 
        description="Test positive sentiment routing",
        nodes=[
            # Prompt tool with positive sentiment text
            NodeSpec(
                id="data_input", 
                type="tool",
                label="Data Input",
                data=NodeData(
                    tool_name="prompt_tool",
                    config={"message": "buy 1000 shares of TSLA now, great opportunity!"}
                )
            ),
            # Decision agent with conditional_gate
            NodeSpec(
                id="decision_agent", 
                type="agent",
                label="Decision Agent",
                data=NodeData(
                    agent_instructions="""Analyze the input text for trading sentiment. Then use conditional_gate to route based on detected sentiment.
                    
                    First parse the message to extract sentiment ("positive" if contains buy/opportunity/bullish, "negative" if contains sell/avoid/risk).
                    Then use conditional_gate with data structure like: {"sentiment": "positive", "message": "original text"}
                    
                    Router config should have:
                    - condition: sentiment field == "positive" → route to 'positive'
                    - condition: sentiment field == "negative" → route to 'negative'  
                    - default route: 'neutral_confirmation'""",
                    tools=["conditional_gate"],
                    model="gpt-4o",
                    sla=SLARequirements(
                        enforce_usage=True,
                        required_tools=["conditional_gate"],
                        tool_usage_required=True
                    )
                )
            ),
            # Positive agent (should execute)
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
            # Negative agent (should be skipped)
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
            # Email agent (should execute with one input from positive_agent using for_each mode)
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
            instructions="""Analyze the input text for trading sentiment. Then use conditional_gate to route based on detected sentiment.
            
            First parse the message to extract sentiment ("positive" if contains buy/opportunity/bullish, "negative" if contains sell/avoid/risk).
            Then use conditional_gate with data structure like: {"sentiment": "positive", "message": "original text"}
            
            Router config should have:
            - condition: sentiment field == "positive" → route to 'positive'
            - condition: sentiment field == "negative" → route to 'negative'  
            - default route: 'neutral_confirmation'""",
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
        conversation_id="test_positive_routing"
    )
    
    # Show execution plan
    print("📋 Execution Plan:")
    for i, batch in enumerate(executor.execution_order):
        print(f"  Batch {i}: {batch}")
    
    # Execute workflow
    initial_state = WorkflowState(
        initial_text="Test positive routing",
        conversation_id="test_positive_routing",
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
        print("\n🔍 POSITIVE ROUTING ANALYSIS:")
        
        # Decision agent should route to positive
        if "decision_agent" in final_state.results:
            decision_result = final_state.results["decision_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    routed_to = getattr(gate_result, 'routed_to', 'unknown')
                    print(f"  ✅ Decision routed to: {routed_to}")
        
        # Check execution status
        positive_executed = "positive_agent" in final_state.results and final_state.results["positive_agent"].get("status") != "skipped"
        negative_skipped = (final_state.results.get("negative_agent", {}).get("status") == "skipped")
        email_executed = "email_agent" in final_state.results and final_state.results["email_agent"].get("status") != "skipped"
        
        print(f"  ✅ positive_agent executed: {positive_executed}")
        print(f"  ✅ negative_agent skipped: {negative_skipped}")
        print(f"  🎯 email_agent executed: {email_executed}")
        
        if positive_executed and negative_skipped and email_executed:
            print("\n🎉 POSITIVE ROUTING WORKING CORRECTLY!")
            print("   positive_agent executed → negative_agent skipped → email_agent got one input")
        else:
            print("\n❌ UNEXPECTED ROUTING BEHAVIOR:")
            print(f"   positive_executed: {positive_executed}")
            print(f"   negative_skipped: {negative_skipped}")
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
    asyncio.run(test_positive_routing())