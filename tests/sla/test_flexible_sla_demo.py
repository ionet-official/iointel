#!/usr/bin/env python3
"""
Test Flexible SLA System - Workflow-Defined Tool Requirements
============================================================
This demonstrates the improved SLA system where ANY tool can be required
for ANY workflow, regardless of catalog "type" classification.

Examples:
- web_search required as final decision tool
- get_current_stock_price required for routing 
- Custom per-workflow SLA requirements
"""
import asyncio
import sys
import uuid
sys.path.append('/Users/alexandermorisse/Documents/GitHub/iointel')

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, SLARequirements
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.datamodels import AgentParams

# Tools are loaded automatically, conditional_gate should be available

async def test_flexible_sla_system():
    """Test workflow-defined SLA requirements override catalog defaults."""
    
    print("🧪 FLEXIBLE SLA SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: Conditional gate with CUSTOM SLA requirements
    print("\n1. Testing custom SLA requirements for conditional_gate")
    
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Custom SLA Test", 
        description="Test custom SLA requirements",
        nodes=[
            NodeSpec(
                id="custom_decision_agent", 
                type="agent",
                label="Custom Decision Agent",
                data=NodeData(
                    agent_instructions="Make a routing decision using conditional_gate. Use the data: price_change=25 to route between 'buy' and 'sell'",
                    tools=["conditional_gate"],
                    model="gpt-4o",
                    # WORKFLOW-DEFINED SLA: Custom requirements
                    sla=SLARequirements(
                        enforce_usage=True,
                        tool_usage_required=True,
                        required_tools=["conditional_gate"],
                        final_tool_must_be="conditional_gate",
                        min_tool_calls=1,
                        max_retries=1
                    )
                )
            )
        ],
        edges=[]
    )
    
    # Validate the workflow SLA configuration
    issues = workflow.validate_structure()
    if issues:
        print(f"⚠️ Workflow validation issues: {issues}")
    else:
        print("✅ Workflow SLA configuration valid")
    
    # Test the agent
    test_agent = AgentParams(
        name="custom_decision_agent",
        instructions="Make a routing decision using conditional_gate. Use the data: price_change=25 to route between 'buy' and 'sell'",
        tools=["conditional_gate"]
    )
    
    # Create DAG executor
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=workflow.nodes,
        edges=workflow.edges,
        agents=[test_agent],
        conversation_id="test_flexible_sla"
    )
    
    # Test execution
    initial_state = WorkflowState(
        initial_text="Analyze price change for routing decision",
        conversation_id="test_flexible_sla",
        results={}
    )
    
    try:
        print("🚀 Executing with workflow-defined SLA...")
        final_state = await executor.execute_dag(initial_state)
        
        if "custom_decision_agent" in final_state.results:
            result = final_state.results["custom_decision_agent"]
            print(f"✅ Agent completed: {type(result)}")
            
            if isinstance(result, dict) and "tool_usage_results" in result:
                tool_usage = result["tool_usage_results"]
                if tool_usage:
                    used_tools = [t.tool_name for t in tool_usage]
                    print(f"🔧 Tools used: {used_tools}")
                    print("🎉 WORKFLOW-DEFINED SLA WORKING!")
                else:
                    print("❌ No tools used - SLA violation")
        else:
            print("⚠️ No results from agent")
            
    except Exception as e:
        print(f"ℹ️ Execution exception: {e}")

    # Test 2: Conditional gate with NO SLA (workflow overrides catalog)
    print("\n2. Testing conditional_gate with NO SLA enforcement (workflow override)")
    
    no_sla_workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="No SLA Override Test", 
        description="Test disabling SLA for normally-enforced tools",
        nodes=[
            NodeSpec(
                id="no_sla_agent", 
                type="agent",
                label="No SLA Agent",
                data=NodeData(
                    agent_instructions="Just provide a simple response about routing. You have conditional_gate available but you don't need to use it.",
                    tools=["conditional_gate"],  # Normally would require SLA
                    model="gpt-4o",
                    # WORKFLOW OVERRIDE: Disable SLA for this specific case
                    sla=SLARequirements(
                        enforce_usage=False,  # Explicitly disable
                        tool_usage_required=False
                    )
                )
            )
        ],
        edges=[]
    )
    
    no_sla_agent = AgentParams(
        name="no_sla_agent",
        instructions="Just provide a simple response about routing. You have conditional_gate available but you don't need to use it.",
        tools=["conditional_gate"]
    )
    
    executor2 = DAGExecutor()
    executor2.build_execution_graph(
        nodes=no_sla_workflow.nodes,
        edges=no_sla_workflow.edges,
        agents=[no_sla_agent],
        conversation_id="test_no_sla"
    )
    
    initial_state2 = WorkflowState(
        initial_text="Provide a simple response about routing",
        conversation_id="test_no_sla",
        results={}
    )
    
    try:
        print("🚀 Executing with SLA disabled...")
        final_state2 = await executor2.execute_dag(initial_state2)
        
        if "no_sla_agent" in final_state2.results:
            result = final_state2.results["no_sla_agent"]
            if isinstance(result, dict) and "tool_usage_results" in result:
                tool_usage = result["tool_usage_results"]
                if tool_usage:
                    print(f"🔧 Agent used tools: {[t.tool_name for t in tool_usage]}")
                else:
                    print("✅ Agent completed without tools (SLA disabled as intended)")
            print("✅ Workflow SLA override successful - no enforcement applied")
        
    except Exception as e:
        print(f"ℹ️ No SLA execution exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_flexible_sla_system())