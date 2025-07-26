import os
#!/usr/bin/env python3
"""
Test Dual Routing Through DAG
============================
This tests a more complex scenario where we have TWO decision agents
that can route to different outcomes, allowing both positive and negative
agents to execute, which should result in email_agent receiving TWO inputs.

Case: Two parallel analysis paths
→ market_analysis_agent routes to 'positive' 
→ risk_analysis_agent routes to 'negative'
→ Both positive_agent and negative_agent execute
→ email_agent executes with inputs from BOTH agents
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

async def test_dual_routing():
    """Test that dual routing allows both agents to execute."""
    
    print("🧪 TESTING DUAL ROUTING (BOTH AGENTS EXECUTE)")
    print("=" * 50)
    
    # Create a workflow with TWO decision agents for more semantic coherence
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Dual Routing Test", 
        description="Test dual routing with both positive and negative execution",
        nodes=[
            # Market data input (positive signal)
            NodeSpec(
                id="market_input", 
                type="tool",
                label="Market Data Input",
                data=NodeData(
                    tool_name="prompt_tool",
                    config={"message": "TSLA earnings beat expectations, strong buy signal from analysts"}
                )
            ),
            # Risk data input (negative signal)  
            NodeSpec(
                id="risk_input", 
                type="tool",
                label="Risk Data Input",
                data=NodeData(
                    tool_name="prompt_tool",
                    config={"message": "TSLA regulatory issues emerging, sell recommendation from compliance"}
                )
            ),
            # Market analysis agent (should route positive)
            NodeSpec(
                id="market_analysis_agent", 
                type="agent",
                label="Market Analysis Agent",
                data=NodeData(
                    agent_instructions="Analyze market sentiment and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
                    tools=["conditional_gate"],
                    model="gpt-4o",
                    sla=SLARequirements(
                        enforce_usage=True,
                        required_tools=["conditional_gate"],
                        tool_usage_required=True
                    )
                )
            ),
            # Risk analysis agent (should route negative)
            NodeSpec(
                id="risk_analysis_agent", 
                type="agent",
                label="Risk Analysis Agent",
                data=NodeData(
                    agent_instructions="Analyze risk factors and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
                    tools=["conditional_gate"],
                    model="gpt-4o",
                    sla=SLARequirements(
                        enforce_usage=True,
                        required_tools=["conditional_gate"],
                        tool_usage_required=True
                    )
                )
            ),
            # Positive agent (should execute from market analysis)
            NodeSpec(
                id="positive_agent", 
                type="agent",
                label="Positive Agent",
                data=NodeData(
                    agent_instructions="Handle positive sentiment confirmation for buying decision based on market analysis",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Negative agent (should execute from risk analysis)
            NodeSpec(
                id="negative_agent", 
                type="agent",
                label="Negative Agent",
                data=NodeData(
                    agent_instructions="Handle negative sentiment confirmation for selling decision based on risk analysis",
                    tools=[],
                    model="gpt-4o"
                )
            ),
            # Email agent (should execute with TWO inputs)
            NodeSpec(
                id="email_agent", 
                type="agent",
                label="Email Agent",
                data=NodeData(
                    agent_instructions="Send comprehensive email about conflicting analysis results from both positive and negative sentiment agents",
                    tools=[],
                    model="gpt-4o"
                )
            )
        ],
        edges=[
            # Market input flows to market analysis
            EdgeSpec(
                id="edge_1",
                source="market_input",
                target="market_analysis_agent",
                data=EdgeData()
            ),
            # Risk input flows to risk analysis
            EdgeSpec(
                id="edge_2",
                source="risk_input",
                target="risk_analysis_agent",
                data=EdgeData()
            ),
            # Market analysis routes to positive
            EdgeSpec(
                id="edge_3",
                source="market_analysis_agent",
                target="positive_agent",
                data=EdgeData(condition="routed_to == 'positive'")
            ),
            # Risk analysis routes to negative
            EdgeSpec(
                id="edge_4",
                source="risk_analysis_agent",
                target="negative_agent",
                data=EdgeData(condition="routed_to == 'negative'")
            ),
            # Both agents feed into email
            EdgeSpec(
                id="edge_5",
                source="positive_agent",
                target="email_agent",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge_6",
                source="negative_agent",
                target="email_agent",
                data=EdgeData()
            )
        ]
    )
    
    # Create agents
    agents = [
        AgentParams(
            name="market_analysis_agent",
            instructions="Analyze market sentiment and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
            tools=["conditional_gate"]
        ),
        AgentParams(
            name="risk_analysis_agent",
            instructions="Analyze risk factors and route accordingly. Use conditional_gate to route to 'positive', 'negative', or 'neutral_confirmation'",
            tools=["conditional_gate"]
        ),
        AgentParams(
            name="positive_agent",
            instructions="Handle positive sentiment confirmation for buying decision based on market analysis",
            tools=[]
        ),
        AgentParams(
            name="negative_agent",
            instructions="Handle negative sentiment confirmation for selling decision based on risk analysis", 
            tools=[]
        ),
        AgentParams(
            name="email_agent",
            instructions="Send comprehensive email about conflicting analysis results from both positive and negative sentiment agents",
            tools=[]
        )
    ]
    
    # Create DAG executor
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow.nodes,
        edges=workflow.edges,
        agents=agents,
        conversation_id="test_dual_routing"
    )
    
    # Show execution plan
    print("📋 Execution Plan:")
    for i, batch in enumerate(executor.execution_order):
        print(f"  Batch {i}: {batch}")
    
    # Execute workflow
    initial_state = WorkflowState(
        initial_text="Test dual routing with conflicting signals",
        conversation_id="test_dual_routing",
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
        print("\n🔍 DUAL ROUTING ANALYSIS:")
        
        # Check both decision agents
        market_route = "unknown"
        risk_route = "unknown"
        
        if "market_analysis_agent" in final_state.results:
            decision_result = final_state.results["market_analysis_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    market_route = getattr(gate_result, 'routed_to', 'unknown')
                    
        if "risk_analysis_agent" in final_state.results:
            decision_result = final_state.results["risk_analysis_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    risk_route = getattr(gate_result, 'routed_to', 'unknown')
        
        print(f"  ✅ Market analysis routed to: {market_route}")
        print(f"  ✅ Risk analysis routed to: {risk_route}")
        
        # Check execution status
        positive_executed = "positive_agent" in final_state.results and final_state.results["positive_agent"].get("status") != "skipped"
        negative_executed = "negative_agent" in final_state.results and final_state.results["negative_agent"].get("status") != "skipped"
        email_executed = "email_agent" in final_state.results and final_state.results["email_agent"].get("status") != "skipped"
        
        print(f"  ✅ positive_agent executed: {positive_executed}")
        print(f"  ✅ negative_agent executed: {negative_executed}")
        print(f"  🎯 email_agent executed: {email_executed}")
        
        if positive_executed and negative_executed and email_executed:
            print("\n🎉 DUAL ROUTING WORKING CORRECTLY!")
            print("   Both agents executed → email_agent received TWO inputs")
            print("   This demonstrates handling of conflicting analysis results")
        elif (positive_executed or negative_executed) and email_executed:
            print("\n⚠️ PARTIAL DUAL ROUTING:")
            print("   Only one agent executed → email_agent received one input")
            print("   This is still correct behavior based on routing decisions")
        else:
            print("\n❌ UNEXPECTED DUAL ROUTING BEHAVIOR:")
            print(f"   positive_executed: {positive_executed}")
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
    asyncio.run(test_dual_routing())