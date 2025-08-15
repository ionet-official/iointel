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
import os
import sys
import uuid
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec, EdgeData, SLARequirements
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.utilities.io_logger import get_component_logger

# Import tools to register them
from iointel.src.agent_methods.tools.user_input import prompt_tool
from iointel.src.agent_methods.tools import conditional_gate
import os

# Create beautiful IOLogger for test output
test_logger = get_component_logger("ROUTING_TEST")

async def test_positive_routing():
    """Test that positive routing works correctly."""
    
    test_logger.info("🧪 TESTING POSITIVE SENTIMENT ROUTING", data={
        "test_type": "positive_sentiment_routing",
        "description": "Testing conditional routing with positive sentiment input",
        "expected_flow": ["data_input", "decision_agent", "positive_agent", "email_agent"],
        "skipped_nodes": ["negative_agent"]
    })
    
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
                type="data_source",
                label="Data Input",
                data=NodeData(
                    source_name="prompt_tool",
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
    
    # Show execution plan with beautiful logging
    test_logger.info("📋 DAG Execution Plan Generated", data={
        "total_batches": len(executor.execution_order),
        "execution_batches": {f"batch_{i}": batch for i, batch in enumerate(executor.execution_order)},
        "parallelizable_nodes": sum(len(batch) > 1 for batch in executor.execution_order)
    })
    
    # Execute workflow
    initial_state = WorkflowState(
        initial_text="Test positive routing",
        conversation_id="test_positive_routing",
        results={}
    )
    
    try:
        test_logger.info("🚀 Starting workflow execution", data={
            "initial_state": "test_positive_routing",
            "input_message": "buy 1000 shares of TSLA now, great opportunity!",
            "expected_routing": "positive"
        })
        final_state = await executor.execute_dag(initial_state)
        
        # Analyze execution results with beautiful logging
        execution_results = {}
        for node_id, result in final_state.results.items():
            if isinstance(result, dict) and "status" in result:
                execution_results[node_id] = {
                    "status": result["status"],
                    "reason": result.get("reason", ""),
                    "executed": result["status"] != "skipped"
                }
            else:
                execution_results[node_id] = {
                    "status": "completed",
                    "reason": "",
                    "executed": True
                }
        
        test_logger.success("📊 Workflow execution completed", data={
            "execution_results": execution_results,
            "total_nodes": len(final_state.results),
            "executed_nodes": sum(1 for r in execution_results.values() if r["executed"]),
            "skipped_nodes": sum(1 for r in execution_results.values() if not r["executed"])
        })
        
        # Analyze routing behavior with detailed logging
        routing_analysis = {"decision_route": "unknown", "gate_confidence": 0.0, "audit_trail": {}}
        
        if "decision_agent" in final_state.results:
            decision_result = final_state.results["decision_agent"]
            if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
                tool_usage = decision_result["tool_usage_results"]
                if tool_usage:
                    gate_result = tool_usage[0].tool_result
                    routing_analysis.update({
                        "decision_route": getattr(gate_result, 'routed_to', 'unknown'),
                        "gate_confidence": getattr(gate_result, 'confidence', 0.0),
                        "decision_reason": getattr(gate_result, 'decision_reason', ''),
                        "audit_trail": getattr(gate_result, 'audit_trail', {})
                    })
        
        # Validate routing behavior
        positive_executed = "positive_agent" in final_state.results and final_state.results["positive_agent"].get("status") != "skipped"
        negative_skipped = (final_state.results.get("negative_agent", {}).get("status") == "skipped")
        email_executed = "email_agent" in final_state.results and final_state.results["email_agent"].get("status") != "skipped"
        
        routing_validation = {
            "positive_agent_executed": positive_executed,
            "negative_agent_skipped": negative_skipped,
            "email_agent_executed": email_executed,
            "routing_correct": positive_executed and negative_skipped and email_executed
        }
        
        test_logger.info("🔍 Routing behavior analysis", data={
            **routing_analysis,
            "validation_results": routing_validation
        })
        
        # Final test results
        stats = executor.get_execution_statistics()
        if routing_validation["routing_correct"]:
            test_logger.success("🎉 POSITIVE ROUTING TEST PASSED", data={
                "test_result": "SUCCESS",
                "flow_validation": "positive_agent executed → negative_agent skipped → email_agent processed one input",
                "execution_stats": stats,
                "routing_efficiency": f"{stats['executed_nodes']}/{stats['total_nodes']} nodes executed"
            })
        else:
            test_logger.error("❌ POSITIVE ROUTING TEST FAILED", data={
                "test_result": "FAILURE",
                "validation_failures": routing_validation,
                "execution_stats": stats
            })
        
    except Exception as e:
        test_logger.critical("❌ Test execution failed", data={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "test_phase": "workflow_execution"
        })
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_positive_routing())