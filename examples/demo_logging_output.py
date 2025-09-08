#!/usr/bin/env python3
"""
Demo script to show the beautiful IO.net logging system in action.

Run this to see what the logging output looks like when workflows execute.
"""

import os
import sys
import time
from uuid import uuid4

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iointel.src.utilities.io_logger import (
    execution_logger, 
    workflow_logger, 
    agent_logger, 
    system_logger
)
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, 
    NodeSpec, 
    NodeData, 
    EdgeSpec, 
    EdgeData,
    SLARequirements
)


def create_demo_workflow():
    """Create a demo workflow for logging demonstration."""
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Enhanced Stock Trading with IO.net Intelligence",
        description="Automated trading workflow with conditional routing and SLA enforcement",
        nodes=[
            NodeSpec(
                id="market_analyzer",
                type="decision",
                label="Market Intelligence Agent",
                data=NodeData(
                    agent_instructions="Analyze market conditions using multiple data sources and AI models",
                    tools=["get_current_stock_price", "conditional_gate", "market_sentiment_analyzer"],
                    sla=SLARequirements(
                        enforce_usage=True,
                        final_tool_must_be="conditional_gate",
                        min_tool_calls=2,
                        required_tools=["get_current_stock_price"]
                    )
                )
            ),
            NodeSpec(
                id="buy_executor",
                type="agent",
                label="Buy Order Execution Agent",
                data=NodeData(
                    agent_instructions="Execute optimized buy orders with risk management",
                    tools=["place_buy_order", "calculate_position_size", "send_notification"]
                )
            ),
            NodeSpec(
                id="sell_executor",
                type="agent",
                label="Sell Order Execution Agent", 
                data=NodeData(
                    agent_instructions="Execute sell orders with profit optimization",
                    tools=["place_sell_order", "calculate_profits", "send_notification"]
                )
            ),
            NodeSpec(
                id="hold_monitor",
                type="agent",
                label="Portfolio Monitoring Agent",
                data=NodeData(
                    agent_instructions="Monitor portfolio and set alerts for future actions",
                    tools=["update_portfolio", "set_price_alerts", "generate_report"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="buy_edge",
                source="market_analyzer",
                target="buy_executor",
                data=EdgeData(condition="routed_to == 'buy'")
            ),
            EdgeSpec(
                id="sell_edge", 
                source="market_analyzer",
                target="sell_executor",
                data=EdgeData(condition="routed_to == 'sell'")
            ),
            EdgeSpec(
                id="hold_edge",
                source="market_analyzer", 
                target="hold_monitor",
                data=EdgeData(condition="routed_to == 'hold'")
            )
        ]
    )


def simulate_workflow_execution():
    """Simulate a complete workflow execution with beautiful logging."""
    
    print("🚀 Starting IO.net Workflow Execution Demo")
    print("=" * 50)
    print()
    
    # Create demo workflow
    workflow = create_demo_workflow()
    execution_id = str(uuid4())[:8]
    
    # 1. Workflow Initialization
    workflow_logger.info(
        "Workflow initialization started",
        data={
            "workflow_title": workflow.title,
            "total_nodes": len(workflow.nodes),
            "total_edges": len(workflow.edges),
            "decision_nodes": 1,
            "sla_enforced_nodes": 1
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 2. Workflow Validation
    workflow_logger.success(
        "Workflow validation completed",
        data={
            "validation_issues": 0,
            "routing_tools_detected": ["conditional_gate"],
            "sla_requirements_validated": True
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 3. Execution Start
    execution_logger.info(
        "Workflow execution started", 
        data={
            "user_input": "TSLA stock analysis",
            "execution_mode": "conditional_routing",
            "timestamp": "2025-07-22T15:45:30Z"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 4. Node Execution - Market Analyzer
    agent_logger.info(
        "Node execution started: Market Intelligence Agent",
        data={
            "node_id": "market_analyzer",
            "available_tools": 3,
            "sla_requirements": ["min_tool_calls: 2", "final_tool_must_be: conditional_gate"]
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Tool executed: get_current_stock_price",
        data={
            "tool_result": {"symbol": "TSLA", "price": 245.67, "change": "+2.3%"},
            "execution_time": 1.2,
            "sla_progress": "1/2 required tools completed"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Tool executed: market_sentiment_analyzer", 
        data={
            "sentiment_score": 0.78,
            "market_indicators": ["bullish", "high_volume", "positive_news"],
            "confidence": 0.85
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Tool executed: conditional_gate",
        data={
            "routing_decision": "sell",
            "confidence": 1.0,
            "reason": "Price up 2.3% with high sentiment - take profits",
            "routed_to": "sell_executor"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Node completed: Market Intelligence Agent",
        data={
            "node_id": "market_analyzer", 
            "execution_time": 3.2,
            "tools_used": 3,
            "sla_compliance": "✓ PASSED",
            "routing_result": "sell"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 5. Conditional Routing
    execution_logger.info(
        "Conditional routing executed",
        data={
            "routing_decision": "sell",
            "target_node": "sell_executor", 
            "skipped_nodes": ["buy_executor", "hold_monitor"],
            "routing_efficiency": "1/3 nodes executed (optimal for conditional)"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 6. Sell Agent Execution  
    agent_logger.info(
        "Node execution started: Sell Order Execution Agent",
        data={
            "node_id": "sell_executor",
            "inherited_data": {"stock": "TSLA", "current_price": 245.67},
            "available_tools": 3
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Tool executed: calculate_profits",
        data={
            "purchase_price": 220.45,
            "current_price": 245.67,
            "profit_amount": 2522.00,
            "profit_percentage": 11.4
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Tool executed: place_sell_order",
        data={
            "order_id": "SELL_TSLA_001",
            "quantity": 100,
            "price": 245.67,
            "order_type": "market",
            "status": "filled"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.3)
    
    agent_logger.success(
        "Node completed: Sell Order Execution Agent",
        data={
            "node_id": "sell_executor",
            "execution_time": 2.1,
            "order_status": "completed",
            "profit_realized": 2522.00
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 7. Execution Completion
    execution_logger.success(
        "Workflow execution completed",
        data={
            "execution_id": execution_id,
            "workflow_title": workflow.title,
            "total_results": 2,
            "execution_time": 8.3,
            "nodes_executed": 2,
            "nodes_skipped": 2,  
            "status": "completed",
            "conditional_routing_success": True,
            "final_result": "Successfully sold 100 TSLA shares for $2,522 profit"
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 8. WorkflowPlanner Analysis Report
    execution_logger.execution_report(
        f"WorkflowPlanner Analysis for {workflow.title}",
        report_data={
            "execution_id": execution_id,
            "status": "completed",
            "duration": "8.3s",
            "nodes_executed": 2,
            "nodes_skipped": 2,
            "workflow_spec": workflow,
            "execution_summary": "Conditional routing worked perfectly. Market analyzer correctly identified sell opportunity and routed to sell agent. SLA compliance achieved.",
            "feedback_prompt": f"Generated comprehensive feedback prompt for {workflow.title} including WORKFLOW SPECIFICATION with topology, SLAs, and EXECUTION RESULTS showing successful conditional routing with EXPECTED EXECUTION PATTERNS validation."
        },
        execution_id=execution_id
    )
    
    time.sleep(0.5)
    
    # 9. Analysis Completion
    execution_logger.success(
        "WorkflowPlanner analysis completed",
        data={
            "analysis_length": 156,
            "analysis_preview": "Conditional routing worked perfectly. Market analyzer correctly identified sell opportunity and routed to sell agent. SLA compliance achieved.",
            "has_workflow_spec": True,
            "feedback_conversation_id": f"feedback_{execution_id}",
            "routing_analysis": "✓ CORRECT - Only sell path executed as expected",
            "sla_analysis": "✓ COMPLIANT - All requirements met"
        },
        execution_id=execution_id
    )
    
    # 10. Final System Summary
    system_logger.success(
        "🎯 IO.net Workflow Engine execution completed successfully!",
        data={
            "total_execution_time": "8.3s",
            "workflow_efficiency": "100% (optimal conditional routing)", 
            "sla_compliance_rate": "100%",
            "profit_generated": "$2,522",
            "next_recommended_action": "Monitor TSLA for re-entry opportunities"
        },
        execution_id=execution_id
    )
    
    print()
    print("🔥 Demo completed! This is what you'll see in your execution logs.")
    print("=" * 60)


if __name__ == "__main__":
    simulate_workflow_execution()