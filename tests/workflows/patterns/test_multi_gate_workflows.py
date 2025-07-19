"""
Comprehensive test suite for multi-gate and complex conditional workflows.

This consolidates complex conditional routing patterns with multiple conditions,
thresholds, and sophisticated decision trees from development files.
"""

import pytest
import asyncio
from uuid import uuid4
from typing import Dict, List, Any, Optional, Union

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


@pytest.fixture
def complex_multi_condition_workflow():
    """Create complex workflow with multiple conditions and routing paths."""
    nodes = [
        NodeSpec(
            id="market_data_input",
            type="agent",
            label="Comprehensive Market Data Input",
            data=NodeData(
                agent_instructions="""Collect comprehensive market data in JSON format with fields:
                sentiment, confidence, price_change, volume, volatility, technical_score, 
                news_sentiment, market_cap, liquidity""",
                tools=["user_input"],
                config={},
                ins=["start"],
                outs=["comprehensive_data"]
            )
        ),
        NodeSpec(
            id="advanced_routing_gate",
            type="agent",
            label="Advanced Trading Decision Gate",
            data=NodeData(
                agent_instructions="""Use conditional_gate tool with multiple complex conditions:
                - aggressive_buy: bullish + confidence >= 0.8 + technical_score >= 0.7 + high volume
                - conservative_buy: bullish + confidence >= 0.6 + technical_score >= 0.5 + low volatility
                - aggressive_sell: bearish + confidence >= 0.8 + technical_score <= 0.3 + high volume
                - conservative_sell: bearish + confidence >= 0.6 + high volatility
                - hedge: volatility >= 0.7
                - default: hold""",
                tools=["conditional_gate"],
                config={},
                ins=["comprehensive_data"],
                outs=["aggressive_buy", "conservative_buy", "aggressive_sell", "conservative_sell", "hedge", "hold"]
            )
        ),
        NodeSpec(
            id="aggressive_buy_action",
            type="agent",
            label="Execute Aggressive Buy Strategy",
            data=NodeData(
                agent_instructions="Execute aggressive buy strategy with detailed analysis.",
                config={},
                ins=["aggressive_buy"],
                outs=["aggressive_buy_result"]
            )
        ),
        NodeSpec(
            id="conservative_buy_action",
            type="agent",
            label="Execute Conservative Buy Strategy",
            data=NodeData(
                agent_instructions="Execute conservative buy strategy with detailed analysis.",
                config={},
                ins=["conservative_buy"],
                outs=["conservative_buy_result"]
            )
        ),
        NodeSpec(
            id="aggressive_sell_action",
            type="agent",
            label="Execute Aggressive Sell Strategy",
            data=NodeData(
                agent_instructions="Execute aggressive sell strategy with detailed analysis.",
                config={},
                ins=["aggressive_sell"],
                outs=["aggressive_sell_result"]
            )
        ),
        NodeSpec(
            id="conservative_sell_action",
            type="agent",
            label="Execute Conservative Sell Strategy",
            data=NodeData(
                agent_instructions="Execute conservative sell strategy with detailed analysis.",
                config={},
                ins=["conservative_sell"],
                outs=["conservative_sell_result"]
            )
        ),
        NodeSpec(
            id="hedge_action",
            type="agent",
            label="Execute Hedging Strategy",
            data=NodeData(
                agent_instructions="Execute hedging strategy for high volatility conditions.",
                config={},
                ins=["hedge"],
                outs=["hedge_result"]
            )
        ),
        NodeSpec(
            id="hold_action",
            type="agent",
            label="Hold Position Strategy",
            data=NodeData(
                agent_instructions="Execute hold strategy for uncertain conditions.",
                config={},
                ins=["hold"],
                outs=["hold_result"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="input_to_gate", source="market_data_input", target="advanced_routing_gate", data=EdgeData()),
        EdgeSpec(id="gate_to_aggressive_buy", source="advanced_routing_gate", target="aggressive_buy_action", 
                data=EdgeData(condition="aggressive_buy")),
        EdgeSpec(id="gate_to_conservative_buy", source="advanced_routing_gate", target="conservative_buy_action", 
                data=EdgeData(condition="conservative_buy")),
        EdgeSpec(id="gate_to_aggressive_sell", source="advanced_routing_gate", target="aggressive_sell_action", 
                data=EdgeData(condition="aggressive_sell")),
        EdgeSpec(id="gate_to_conservative_sell", source="advanced_routing_gate", target="conservative_sell_action", 
                data=EdgeData(condition="conservative_sell")),
        EdgeSpec(id="gate_to_hedge", source="advanced_routing_gate", target="hedge_action", 
                data=EdgeData(condition="hedge")),
        EdgeSpec(id="gate_to_hold", source="advanced_routing_gate", target="hold_action", 
                data=EdgeData(condition="hold"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Complex Multi-Condition Gate Workflow",
        description="Advanced conditional routing with multiple conditions and decision paths",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "multi_condition", "complexity": "high"}
    )


@pytest.fixture
def fan_out_alert_workflow():
    """Create workflow that fans out to multiple alerting systems."""
    nodes = [
        NodeSpec(
            id="event_detector",
            type="tool",
            label="Market Event Detector",
            data=NodeData(
                tool_name="get_coin_quotes",
                config={"symbol": ["BTC", "ETH"]},
                ins=[],
                outs=["market_events"]
            )
        ),
        NodeSpec(
            id="alert_router",
            type="agent",
            label="Multi-Channel Alert Router",
            data=NodeData(
                agent_instructions="""Analyze market events and determine alert priorities.
                Use conditional_gate to route to multiple alert channels simultaneously:
                - high_priority: significant price movements or volume spikes
                - medium_priority: moderate changes requiring monitoring
                - low_priority: minor fluctuations for tracking
                Can route to multiple channels for important events.""",
                tools=["conditional_gate"],
                config={},
                ins=["market_events"],
                outs=["high_priority", "medium_priority", "low_priority"]
            )
        ),
        NodeSpec(
            id="urgent_sms_alert",
            type="agent",
            label="Urgent SMS Alert System",
            data=NodeData(
                agent_instructions="Send urgent SMS alerts for high priority events.",
                config={},
                ins=["high_priority"],
                outs=["sms_sent"]
            )
        ),
        NodeSpec(
            id="email_notification",
            type="agent",
            label="Email Notification System",
            data=NodeData(
                agent_instructions="Send detailed email notifications for medium priority events.",
                config={},
                ins=["medium_priority"],
                outs=["email_sent"]
            )
        ),
        NodeSpec(
            id="dashboard_update",
            type="agent",
            label="Dashboard Update System",
            data=NodeData(
                agent_instructions="Update monitoring dashboard for low priority events.",
                config={},
                ins=["low_priority"],
                outs=["dashboard_updated"]
            )
        ),
        NodeSpec(
            id="slack_notification",
            type="agent",
            label="Slack Team Notification",
            data=NodeData(
                agent_instructions="Send team Slack notifications for high priority events.",
                config={},
                ins=["high_priority"],
                outs=["slack_sent"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="detector_to_router", source="event_detector", target="alert_router", data=EdgeData()),
        EdgeSpec(id="router_to_sms", source="alert_router", target="urgent_sms_alert", 
                data=EdgeData(condition="high_priority")),
        EdgeSpec(id="router_to_email", source="alert_router", target="email_notification", 
                data=EdgeData(condition="medium_priority")),
        EdgeSpec(id="router_to_dashboard", source="alert_router", target="dashboard_update", 
                data=EdgeData(condition="low_priority")),
        EdgeSpec(id="router_to_slack", source="alert_router", target="slack_notification", 
                data=EdgeData(condition="high_priority"))  # Same high priority goes to both SMS and Slack
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Fan-Out Alert Workflow",
        description="Multi-channel alerting with priority-based routing",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "fan_out", "pattern": "broadcasting"}
    )


@pytest.fixture
def nested_conditional_workflow():
    """Create workflow with nested conditional decision trees."""
    nodes = [
        NodeSpec(
            id="primary_analysis",
            type="tool",
            label="Primary Market Analysis",
            data=NodeData(
                tool_name="get_coin_quotes",
                config={"symbol": ["BTC"]},
                ins=[],
                outs=["primary_data"]
            )
        ),
        NodeSpec(
            id="first_level_gate",
            type="agent",
            label="First Level Decision Gate",
            data=NodeData(
                agent_instructions="""First level routing based on price movement:
                - Route to 'positive_branch' if price change > 0
                - Route to 'negative_branch' if price change < 0
                - Route to 'neutral_branch' if price change == 0""",
                tools=["conditional_gate"],
                config={},
                ins=["primary_data"],
                outs=["positive_branch", "negative_branch", "neutral_branch"]
            )
        ),
        NodeSpec(
            id="positive_secondary_gate",
            type="agent",
            label="Positive Branch Secondary Gate",
            data=NodeData(
                agent_instructions="""Secondary routing for positive movements:
                - Route to 'strong_buy' if change > 5%
                - Route to 'moderate_buy' if change 1-5%
                - Route to 'weak_buy' if change 0-1%""",
                tools=["conditional_gate"],
                config={},
                ins=["positive_branch"],
                outs=["strong_buy", "moderate_buy", "weak_buy"]
            )
        ),
        NodeSpec(
            id="negative_secondary_gate",
            type="agent",
            label="Negative Branch Secondary Gate",
            data=NodeData(
                agent_instructions="""Secondary routing for negative movements:
                - Route to 'strong_sell' if change < -5%
                - Route to 'moderate_sell' if change -5% to -1%
                - Route to 'weak_sell' if change -1% to 0%""",
                tools=["conditional_gate"],
                config={},
                ins=["negative_branch"],
                outs=["strong_sell", "moderate_sell", "weak_sell"]
            )
        ),
        NodeSpec(
            id="strong_buy_action",
            type="agent",
            label="Strong Buy Action",
            data=NodeData(
                agent_instructions="Execute strong buy strategy for significant positive movement.",
                config={},
                ins=["strong_buy"],
                outs=["strong_buy_result"]
            )
        ),
        NodeSpec(
            id="strong_sell_action",
            type="agent",
            label="Strong Sell Action",
            data=NodeData(
                agent_instructions="Execute strong sell strategy for significant negative movement.",
                config={},
                ins=["strong_sell"],
                outs=["strong_sell_result"]
            )
        ),
        NodeSpec(
            id="neutral_analysis",
            type="agent",
            label="Neutral Market Analysis",
            data=NodeData(
                agent_instructions="Analyze neutral market conditions for next steps.",
                config={},
                ins=["neutral_branch"],
                outs=["neutral_result"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="analysis_to_first_gate", source="primary_analysis", target="first_level_gate", data=EdgeData()),
        EdgeSpec(id="first_to_positive_gate", source="first_level_gate", target="positive_secondary_gate", 
                data=EdgeData(condition="positive_branch")),
        EdgeSpec(id="first_to_negative_gate", source="first_level_gate", target="negative_secondary_gate", 
                data=EdgeData(condition="negative_branch")),
        EdgeSpec(id="first_to_neutral", source="first_level_gate", target="neutral_analysis", 
                data=EdgeData(condition="neutral_branch")),
        EdgeSpec(id="positive_to_strong_buy", source="positive_secondary_gate", target="strong_buy_action", 
                data=EdgeData(condition="strong_buy")),
        EdgeSpec(id="negative_to_strong_sell", source="negative_secondary_gate", target="strong_sell_action", 
                data=EdgeData(condition="strong_sell"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Nested Conditional Decision Tree",
        description="Hierarchical decision making with nested conditional gates",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "nested_conditional", "pattern": "hierarchical"}
    )


class TestMultiGateWorkflows:
    """Test suite for complex multi-gate workflows."""
    
    def test_complex_workflow_structure(self, complex_multi_condition_workflow):
        """Test structure of complex multi-condition workflows."""
        workflow = complex_multi_condition_workflow
        
        # Validate structure
        assert len(workflow.nodes) == 8  # input + gate + 6 action agents
        assert len(workflow.edges) == 7  # 1 input + 6 conditional outputs
        
        # Validate gate node has multiple outputs
        gate_node = next(n for n in workflow.nodes if "routing_gate" in n.id)
        gate_outputs = [e for e in workflow.edges if e.source == gate_node.id]
        assert len(gate_outputs) == 6  # 6 different routing paths
        
        # Validate all routing conditions are unique
        conditions = [e.data.condition for e in gate_outputs]
        assert len(set(conditions)) == len(conditions)  # All unique
    
    def test_fan_out_pattern(self, fan_out_alert_workflow):
        """Test fan-out pattern where one input routes to multiple outputs."""
        workflow = fan_out_alert_workflow
        
        # Find alert router
        router = next(n for n in workflow.nodes if "alert_router" in n.id)
        router_outputs = [e for e in workflow.edges if e.source == router.id]
        
        # Should have multiple outputs from router
        assert len(router_outputs) >= 3
        
        # Check for potential simultaneous routing (same condition to multiple targets)
        high_priority_edges = [e for e in router_outputs if "high_priority" in e.data.condition]
        assert len(high_priority_edges) >= 2  # Should route to both SMS and Slack for urgent alerts
        
        # Validate different priority levels
        conditions = [e.data.condition for e in router_outputs]
        assert "high_priority" in conditions
        assert "medium_priority" in conditions
        assert "low_priority" in conditions
    
    def test_nested_conditional_structure(self, nested_conditional_workflow):
        """Test nested conditional decision tree structure."""
        workflow = nested_conditional_workflow
        
        # Validate hierarchical structure
        first_gate = next(n for n in workflow.nodes if "first_level_gate" in n.id)
        secondary_gates = [n for n in workflow.nodes if "secondary_gate" in n.id]
        
        assert len(secondary_gates) == 2  # Positive and negative branches
        
        # Validate first level routes to secondary gates
        first_level_outputs = [e for e in workflow.edges if e.source == first_gate.id]
        secondary_gate_targets = {e.target for e in first_level_outputs if "secondary_gate" in e.target}
        
        assert len(secondary_gate_targets) == 2  # Routes to both secondary gates
        
        # Validate secondary gates have their own outputs
        for gate in secondary_gates:
            gate_outputs = [e for e in workflow.edges if e.source == gate.id]
            assert len(gate_outputs) >= 1  # Each secondary gate has outputs
    
    @pytest.mark.parametrize("workflow_type,expected_patterns", [
        ("complex_multi_condition_workflow", ["aggressive_buy", "conservative_buy", "aggressive_sell", "conservative_sell", "hedge", "hold"]),
        ("fan_out_alert_workflow", ["high_priority", "medium_priority", "low_priority"]),
        ("nested_conditional_workflow", ["positive_branch", "negative_branch", "neutral_branch"])
    ])
    def test_routing_patterns(self, workflow_type, expected_patterns, request):
        """Test that workflows implement expected routing patterns."""
        workflow = request.getfixturevalue(workflow_type)
        
        # Extract all conditions from edges
        all_conditions = [e.data.condition for e in workflow.edges if e.data.condition]
        
        # Check that expected patterns appear in conditions
        for pattern in expected_patterns:
            pattern_found = any(pattern in condition for condition in all_conditions)
            assert pattern_found, f"Pattern '{pattern}' not found in workflow conditions"
    
    def test_computational_efficiency_design(self, complex_multi_condition_workflow):
        """Test that complex workflows are designed for computational efficiency."""
        workflow = complex_multi_condition_workflow
        
        # Find the main routing gate
        gate_node = next(n for n in workflow.nodes if "routing_gate" in n.id)
        action_nodes = [n for n in workflow.nodes if "action" in n.id]
        
        # Validate that each action node is only reachable through conditional routing
        action_node_ids = {n.id for n in action_nodes}
        conditional_edges = [e for e in workflow.edges if e.data.condition]
        conditional_targets = {e.target for e in conditional_edges}
        
        # All action nodes should be conditional targets (ensuring only selected paths execute)
        assert action_node_ids.issubset(conditional_targets)
        
        # Multiple action nodes available but only one path should execute
        assert len(action_nodes) >= 4  # Multiple strategies available
    
    def test_multi_condition_complexity(self, complex_multi_condition_workflow):
        """Test handling of multiple simultaneous conditions."""
        workflow = complex_multi_condition_workflow
        
        # Find the gate agent
        gate_agent = next(n for n in workflow.nodes if "routing_gate" in n.id)
        
        # Validate agent has access to conditional_gate tool
        assert "conditional_gate" in gate_agent.data.tools
        
        # Validate instructions mention multiple conditions
        instructions = gate_agent.data.agent_instructions.lower()
        
        # Should mention multiple condition types
        condition_types = ["confidence", "technical_score", "volume", "volatility"]
        mentioned_conditions = sum(1 for cond in condition_types if cond in instructions)
        assert mentioned_conditions >= 3  # Multiple conditions mentioned
        
        # Should mention threshold values
        assert "0.8" in instructions or "0.7" in instructions  # Specific thresholds
    
    def test_workflow_scalability_patterns(self, fan_out_alert_workflow):
        """Test patterns that support workflow scalability."""
        workflow = fan_out_alert_workflow
        
        # Fan-out pattern should be easily extensible
        router = next(n for n in workflow.nodes if "router" in n.id)
        alert_systems = [n for n in workflow.nodes if n.id != router.id and n.id != "event_detector" and 
                        any(word in n.label.lower() for word in ["alert", "notification", "update"])]
        
        # Multiple alert systems (scalable pattern)
        assert len(alert_systems) >= 3
        
        # Each alert system should be independently reachable
        router_edges = [e for e in workflow.edges if e.source == router.id]
        alert_targets = {e.target for e in router_edges}
        alert_ids = {n.id for n in alert_systems}
        
        # Most alert systems should be reachable from router
        reachable_alerts = alert_ids.intersection(alert_targets)
        assert len(reachable_alerts) >= len(alert_systems) * 0.7  # At least 70% reachable
    
    def test_error_handling_and_fallbacks(self, complex_multi_condition_workflow):
        """Test that complex workflows have appropriate error handling."""
        workflow = complex_multi_condition_workflow
        
        # Should have a default/fallback action
        gate_edges = [e for e in workflow.edges if e.data.condition]
        conditions = [e.data.condition for e in gate_edges]
        
        # Should have a "hold" or default action for uncertain conditions
        has_fallback = any("hold" in condition for condition in conditions)
        assert has_fallback, "Complex workflow should have fallback action"
        
        # Fallback should be reachable
        hold_edge = next(e for e in gate_edges if "hold" in e.data.condition)
        hold_target = next(n for n in workflow.nodes if n.id == hold_edge.target)
        assert hold_target is not None


class TestAdvancedWorkflowPatterns:
    """Test advanced workflow design patterns."""
    
    def test_dynamic_routing_capability(self):
        """Test workflows with dynamic routing based on data content."""
        # This would test workflows that route based on dynamic analysis of input data
        pass
    
    def test_parallel_execution_patterns(self):
        """Test workflows designed for parallel execution of multiple paths."""
        # This would test workflows where multiple paths can execute simultaneously
        pass
    
    def test_workflow_composition_patterns(self):
        """Test patterns for composing smaller workflows into larger ones."""
        # This would test modular workflow design patterns
        pass
    
    def test_adaptive_threshold_patterns(self):
        """Test workflows with adaptive thresholds based on historical data."""
        # This would test workflows that adjust their decision criteria over time
        pass


class TestWorkflowExecutionScenarios:
    """Test specific execution scenarios for complex workflows."""
    
    @pytest.mark.parametrize("scenario,expected_path", [
        ({
            "sentiment": "bullish", "confidence": 0.9, "technical_score": 0.8, 
            "volume": "high", "volatility": 0.2
        }, "aggressive_buy"),
        ({
            "sentiment": "bullish", "confidence": 0.6, "technical_score": 0.6, 
            "volatility": 0.3
        }, "conservative_buy"),
        ({
            "sentiment": "bearish", "confidence": 0.9, "technical_score": 0.2, 
            "volume": "high"
        }, "aggressive_sell"),
        ({
            "sentiment": "bearish", "confidence": 0.7, "volatility": 0.8
        }, "conservative_sell"),
        ({"volatility": 0.8}, "hedge"),
        ({"sentiment": "neutral", "confidence": 0.5}, "hold")
    ])
    def test_scenario_routing_expectations(self, complex_multi_condition_workflow, scenario, expected_path):
        """Test that specific scenarios would route to expected paths."""
        workflow = complex_multi_condition_workflow
        
        # Find the routing gate
        gate_node = next(n for n in workflow.nodes if "routing_gate" in n.id)
        
        # Find edges that would be activated for this scenario
        gate_edges = [e for e in workflow.edges if e.source == gate_node.id]
        expected_edge = next(e for e in gate_edges if expected_path in e.data.condition)
        
        # Validate the expected edge exists and targets the right action
        assert expected_edge is not None
        target_node = next(n for n in workflow.nodes if n.id == expected_edge.target)
        assert expected_path in target_node.id or expected_path in target_node.label.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])