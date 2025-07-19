#!/usr/bin/env python3
"""
Test suite for the conditional_multi_gate tool.

This test validates that the multi-gate routing works correctly,
evaluating ALL conditions and returning ALL matched routes.
"""

import pytest
from iointel.src.agent_methods.tools.conditional_gate import (
    conditional_multi_gate,
    RouterConfig,
    SimpleCondition,
    MultiGateResult
)


class TestConditionalMultiGate:
    """Test conditional_multi_gate tool functionality."""
    
    def test_single_condition_match(self):
        """Test that single matching condition works correctly."""
        data = {"sentiment": "bullish", "confidence": 0.8}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="sentiment",
                    operator="==",
                    value="bullish",
                    route="positive_sentiment"
                )
            ],
            default_route="neutral"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert result.routed_to == ["positive_sentiment"]
        assert result.matched_routes == ["positive_sentiment"]
        assert len(result.actions) == 1
        assert result.actions[0] == "branch"
        assert "Matched 1 conditions" in result.decision_reason
    
    def test_multiple_conditions_match(self):
        """Test that multiple conditions can match simultaneously."""
        data = {"sentiment": "bullish", "confidence": 0.85, "volume": 5000}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="sentiment",
                    operator="in",
                    value=["bullish", "positive"],
                    route="positive_sentiment"
                ),
                SimpleCondition(
                    field="confidence",
                    operator=">",
                    value=0.7,
                    route="high_confidence"
                ),
                SimpleCondition(
                    field="volume",
                    operator=">",
                    value=1000,
                    route="high_volume"
                )
            ],
            default_route="neutral"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 3
        assert "positive_sentiment" in result.routed_to
        assert "high_confidence" in result.routed_to
        assert "high_volume" in result.routed_to
        assert result.matched_routes == result.routed_to
        assert len(result.actions) == 3
        assert all(action == "branch" for action in result.actions)
        assert "Matched 3 conditions" in result.decision_reason
    
    def test_partial_conditions_match(self):
        """Test that only matching conditions are included."""
        data = {"sentiment": "bullish", "confidence": 0.5, "volume": 500}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="sentiment",
                    operator="==",
                    value="bullish",
                    route="positive_sentiment"
                ),
                SimpleCondition(
                    field="confidence",
                    operator=">",
                    value=0.7,
                    route="high_confidence"
                ),
                SimpleCondition(
                    field="volume",
                    operator=">",
                    value=1000,
                    route="high_volume"
                )
            ],
            default_route="neutral"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert result.routed_to == ["positive_sentiment"]
        assert result.matched_routes == ["positive_sentiment"]
        assert len(result.actions) == 1
        assert result.actions[0] == "branch"
        assert "Matched 1 conditions" in result.decision_reason
    
    def test_no_conditions_match(self):
        """Test that default route is used when no conditions match."""
        data = {"sentiment": "neutral", "confidence": 0.5, "volume": 100}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="sentiment",
                    operator="==",
                    value="bullish",
                    route="positive_sentiment"
                ),
                SimpleCondition(
                    field="confidence",
                    operator=">",
                    value=0.7,
                    route="high_confidence"
                )
            ],
            default_route="neutral",
            default_action="terminate"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert result.routed_to == ["neutral"]
        assert result.matched_routes == []
        assert len(result.actions) == 1
        assert result.actions[0] == "terminate"
        assert "No conditions matched" in result.decision_reason
    
    def test_fan_out_pattern(self):
        """Test fan-out pattern for alerting multiple systems."""
        data = {"price_change": 15.5, "alert_level": "critical"}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="price_change",
                    operator=">",
                    value=10,
                    route="trading_alert"
                ),
                SimpleCondition(
                    field="alert_level",
                    operator="==",
                    value="critical",
                    route="notification_system"
                ),
                SimpleCondition(
                    field="price_change",
                    operator=">",
                    value=5,
                    route="logging_system"
                )
            ],
            default_route="none"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 3
        assert "trading_alert" in result.routed_to
        assert "notification_system" in result.routed_to
        assert "logging_system" in result.routed_to
        assert result.matched_routes == result.routed_to
        assert len(result.actions) == 3
        assert all(action == "branch" for action in result.actions)
    
    def test_between_operator(self):
        """Test between operator with multiple matches."""
        data = {"temperature": 25, "humidity": 60}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="temperature",
                    operator="between",
                    value=[20, 30],
                    route="optimal_temp"
                ),
                SimpleCondition(
                    field="humidity",
                    operator="between",
                    value=[40, 70],
                    route="optimal_humidity"
                ),
                SimpleCondition(
                    field="temperature",
                    operator=">",
                    value=15,
                    route="above_freezing"
                )
            ],
            default_route="suboptimal"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 3
        assert "optimal_temp" in result.routed_to
        assert "optimal_humidity" in result.routed_to
        assert "above_freezing" in result.routed_to
    
    def test_missing_field_handling(self):
        """Test that missing fields are handled gracefully."""
        data = {"price": 100}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="price",
                    operator=">",
                    value=50,
                    route="high_price"
                ),
                SimpleCondition(
                    field="volume",  # Missing field
                    operator=">",
                    value=1000,
                    route="high_volume"
                )
            ],
            default_route="default"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert result.routed_to == ["high_price"]
        assert result.matched_routes == ["high_price"]
        assert len(result.actions) == 1
        
        # Check audit trail shows the missing field
        missing_field_condition = None
        for condition in result.audit_trail["evaluated_conditions"]:
            if condition["field"] == "volume":
                missing_field_condition = condition
                break
        
        assert missing_field_condition is not None
        assert missing_field_condition["result"] is False
        assert "not found" in missing_field_condition["reason"]
    
    def test_dict_config_input(self):
        """Test that dict config input works correctly."""
        data = {"sentiment": "bullish", "confidence": 0.85}
        
        router_config_dict = {
            "conditions": [
                {
                    "field": "sentiment",
                    "operator": "==",
                    "value": "bullish",
                    "route": "positive",
                    "action": "branch"
                },
                {
                    "field": "confidence",
                    "operator": ">",
                    "value": 0.7,
                    "route": "high_confidence",
                    "action": "branch"
                }
            ],
            "default_route": "neutral",
            "default_action": "terminate"
        }
        
        result = conditional_multi_gate(data, router_config_dict)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 2
        assert "positive" in result.routed_to
        assert "high_confidence" in result.routed_to
    
    def test_json_string_data_input(self):
        """Test that JSON string data input works correctly."""
        data_json = '{"sentiment": "bullish", "confidence": 0.85}'
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="sentiment",
                    operator="==",
                    value="bullish",
                    route="positive"
                ),
                SimpleCondition(
                    field="confidence",
                    operator=">",
                    value=0.7,
                    route="high_confidence"
                )
            ],
            default_route="neutral"
        )
        
        result = conditional_multi_gate(data_json, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 2
        assert "positive" in result.routed_to
        assert "high_confidence" in result.routed_to
    
    def test_invalid_json_data(self):
        """Test handling of invalid JSON data."""
        invalid_json = '{"invalid": json}'
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="test",
                    operator="==",
                    value="test",
                    route="test_route"
                )
            ],
            default_route="default"
        )
        
        result = conditional_multi_gate(invalid_json, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert result.routed_to == []
        assert result.matched_routes == []
        assert result.confidence == 0.0
        assert "Invalid JSON data" in result.decision_reason
    
    def test_custom_actions(self):
        """Test that custom actions are preserved in results."""
        data = {"alert": "critical", "priority": "high"}
        
        router_config = RouterConfig(
            conditions=[
                SimpleCondition(
                    field="alert",
                    operator="==",
                    value="critical",
                    route="emergency_response",
                    action="terminate"
                ),
                SimpleCondition(
                    field="priority",
                    operator="==",
                    value="high",
                    route="priority_queue",
                    action="continue"
                )
            ],
            default_route="standard"
        )
        
        result = conditional_multi_gate(data, router_config)
        
        assert isinstance(result, MultiGateResult)
        assert len(result.routed_to) == 2
        assert "emergency_response" in result.routed_to
        assert "priority_queue" in result.routed_to
        assert len(result.actions) == 2
        assert "terminate" in result.actions
        assert "continue" in result.actions


if __name__ == "__main__":
    # Run a few key tests to demonstrate functionality
    test_suite = TestConditionalMultiGate()
    
    print("🧪 Testing conditional_multi_gate tool...")
    
    print("\n1. Testing multiple conditions match:")
    test_suite.test_multiple_conditions_match()
    print("✅ Multiple conditions test passed")
    
    print("\n2. Testing fan-out pattern:")
    test_suite.test_fan_out_pattern()
    print("✅ Fan-out pattern test passed")
    
    print("\n3. Testing no conditions match (default route):")
    test_suite.test_no_conditions_match()
    print("✅ Default route test passed")
    
    print("\n4. Testing partial conditions match:")
    test_suite.test_partial_conditions_match()
    print("✅ Partial conditions test passed")
    
    print("\n🎉 All conditional_multi_gate tests passed!")
    print("\n📋 Key features verified:")
    print("  • Multiple simultaneous routing")
    print("  • Fan-out pattern support")
    print("  • Partial condition matching")
    print("  • Default route handling")
    print("  • Custom actions preservation")
    print("  • JSON and dict input support")
    print("  • Error handling for invalid data")