"""
Tests for decision tools used in workflow conditionals.
"""

import pytest
from iointel.src.agent_methods.tools.decision_tools import (
    json_evaluator,
    number_compare,
    string_contains,
    boolean_mux,
    conditional_router,
    threshold_check,
    DecisionResult,
    RouterResult
)


class TestJsonEvaluator:
    """Test JSON evaluation tool."""

    def test_simple_equality(self):
        """Test simple equality check."""
        print("\n=== Testing JSON Evaluator Simple Equality ===")
        data = {"weather": {"condition": "rain", "temperature": 20}}
        print(f"Input data: {data}")
        print("Expression: data['weather']['condition'] == 'rain'")
        
        result = json_evaluator(data, "data['weather']['condition'] == 'rain'")
        print(f"Result: {result.result}")
        print(f"Details: {result.details}")
        print(f"Confidence: {result.confidence}")
        
        assert isinstance(result, DecisionResult)
        assert result.result is True
        assert "evaluated to True" in result.details

    def test_nested_property_access(self):
        """Test nested property access."""
        data = {"status": {"code": 200, "message": "OK"}}
        result = json_evaluator(data, "data['status']['code'] == 200")
        
        assert result.result is True

    def test_json_string_input(self):
        """Test with JSON string input."""
        data_str = '{"temperature": 25, "humidity": 60}'
        result = json_evaluator(data_str, "data['temperature'] > 20")
        
        assert result.result is True

    def test_false_condition(self):
        """Test condition that evaluates to false."""
        data = {"count": 5}
        result = json_evaluator(data, "data['count'] > 10")
        
        assert result.result is False

    def test_invalid_expression(self):
        """Test with invalid/unsafe expression."""
        data = {"test": "value"}
        result = json_evaluator(data, "import os")
        
        assert result.result is False
        assert "Unsafe expression" in result.details

    def test_evaluation_error(self):
        """Test expression that causes evaluation error."""
        data = {"test": "value"}
        result = json_evaluator(data, "data['nonexistent']['key']")
        
        assert result.result is False
        assert "Evaluation error" in result.details


class TestNumberCompare:
    """Test number comparison tool."""

    def test_greater_than(self):
        """Test greater than comparison."""
        result = number_compare(10, ">", 5)
        
        assert isinstance(result, DecisionResult)
        assert result.result is True
        assert "10 > 5 = True" in result.details

    def test_less_than_false(self):
        """Test less than comparison that's false."""
        result = number_compare(10, "<", 5)
        
        assert result.result is False
        assert "10 < 5 = False" in result.details

    def test_equality_with_floats(self):
        """Test equality with floating point numbers."""
        result = number_compare(3.14159, "==", 3.14159)
        
        assert result.result is True

    def test_string_number_conversion(self):
        """Test converting string to number."""
        result = number_compare("42", ">", 40)
        
        assert result.result is True

    def test_invalid_string_conversion(self):
        """Test invalid string conversion."""
        result = number_compare("not_a_number", ">", 5)
        
        assert result.result is False
        assert "Cannot convert" in result.details

    def test_unknown_operator(self):
        """Test unknown operator."""
        result = number_compare(10, "~=", 5)
        
        assert result.result is False
        assert "Unknown operator" in result.details


class TestStringContains:
    """Test string contains tool."""

    def test_simple_contains(self):
        """Test simple substring check."""
        result = string_contains("Hello World", "World")
        
        assert isinstance(result, DecisionResult)
        assert result.result is True
        assert "found" in result.details

    def test_case_insensitive(self):
        """Test case insensitive search."""
        result = string_contains("Hello World", "world", case_sensitive=False)
        
        assert result.result is True

    def test_case_sensitive_false(self):
        """Test case sensitive search that fails."""
        result = string_contains("Hello World", "world", case_sensitive=True)
        
        assert result.result is False

    def test_regex_pattern(self):
        """Test regex pattern matching."""
        result = string_contains("Email: test@example.com", r"\w+@\w+\.\w+", regex=True)
        
        assert result.result is True
        assert "matched" in result.details

    def test_regex_not_found(self):
        """Test regex pattern not found."""
        result = string_contains("No email here", r"\w+@\w+\.\w+", regex=True)
        
        assert result.result is False

    def test_substring_not_found(self):
        """Test substring not found."""
        result = string_contains("Hello World", "xyz")
        
        assert result.result is False
        assert "not found" in result.details


class TestBooleanMux:
    """Test boolean multiplexer tool."""

    def test_true_path(self):
        """Test routing on true condition."""
        result = boolean_mux(True, "success_route", "failure_route")
        
        assert isinstance(result, RouterResult)
        assert result.routed_to == "success_route"
        assert result.route_data["condition"] is True

    def test_false_path(self):
        """Test routing on false condition."""
        result = boolean_mux(False, "success_route", "failure_route")
        
        assert result.routed_to == "failure_route"
        assert result.route_data["condition"] is False

    def test_default_values(self):
        """Test with default values."""
        result = boolean_mux(True)
        
        assert result.routed_to == "true_path"

    def test_custom_values(self):
        """Test with custom routing values."""
        result = boolean_mux(False, "happy_path", "sad_path")
        
        assert result.routed_to == "sad_path"


class TestConditionalRouter:
    """Test conditional router tool."""

    def test_simple_routing(self):
        """Test simple routing based on decision."""
        decision = {"action": "send_email"}
        routes = {
            "send_email": "email_sender",
            "send_sms": "sms_sender",
            "log_only": "logger"
        }
        
        result = conditional_router(decision, routes)
        
        assert isinstance(result, RouterResult)
        assert result.routed_to == "email_sender"
        assert result.matched_condition == "action == send_email"

    def test_nested_decision_path(self):
        """Test routing with nested decision path."""
        decision = {"result": {"type": "warning", "severity": "high"}}
        routes = {"warning": "warning_handler", "error": "error_handler"}
        
        result = conditional_router(decision, routes, decision_path="result.type")
        
        assert result.routed_to == "warning_handler"

    def test_default_route(self):
        """Test fallback to default route."""
        decision = {"action": "unknown_action"}
        routes = {"send_email": "email_sender"}
        
        result = conditional_router(decision, routes, default_route="fallback_handler")
        
        assert result.routed_to == "fallback_handler"

    def test_no_route_found(self):
        """Test when no route is found and no default."""
        decision = {"action": "unknown"}
        routes = {"send_email": "email_sender"}
        
        result = conditional_router(decision, routes)
        
        assert result.routed_to == "no_route"

    def test_json_string_decision(self):
        """Test with JSON string decision."""
        decision_str = '{"status": "success", "code": 200}'
        routes = {"success": "success_handler", "failure": "failure_handler"}
        
        result = conditional_router(decision_str, routes, decision_path="status")
        
        assert result.routed_to == "success_handler"


class TestThresholdCheck:
    """Test threshold checking tool."""

    def test_single_threshold_exceeded(self):
        """Test exceeding a single threshold."""
        thresholds = {"high": 80, "medium": 50, "low": 20}
        
        result = threshold_check(90, thresholds)
        
        assert isinstance(result, RouterResult)
        assert result.routed_to == "high"
        assert result.route_data["exceeded"] is True

    def test_multiple_thresholds(self):
        """Test with multiple thresholds - should pick highest exceeded."""
        thresholds = {"critical": 90, "warning": 70, "info": 30}
        
        result = threshold_check(75, thresholds)
        
        assert result.routed_to == "warning"  # 75 >= 70 but < 90

    def test_no_threshold_exceeded(self):
        """Test when no thresholds are exceeded."""
        thresholds = {"high": 80, "medium": 50}
        
        result = threshold_check(25, thresholds, default_result="normal")
        
        assert result.routed_to == "normal"
        assert result.route_data["exceeded"] is False

    def test_exact_threshold_match(self):
        """Test exact threshold match."""
        thresholds = {"exact": 50}
        
        result = threshold_check(50, thresholds)
        
        assert result.routed_to == "exact"


class TestToolIntegration:
    """Test tools working together in workflow scenarios."""

    def test_weather_decision_workflow(self):
        """Test a weather-based decision workflow."""
        print("\n=== Testing Weather Decision Workflow ===")
        
        # 1. Evaluate weather condition
        weather_data = {
            "weather": {
                "condition": "rain",
                "temperature": 15,
                "humidity": 85
            }
        }
        print(f"Weather data: {weather_data}")
        
        # Check if it's raining
        print("\nStep 1: Check if raining")
        rain_check = json_evaluator(weather_data, "data['weather']['condition'] == 'rain'")
        print(f"Rain check result: {rain_check.result} - {rain_check.details}")
        assert rain_check.result is True
        
        # Check temperature threshold
        print("\nStep 2: Check temperature threshold")
        temp_check = number_compare(weather_data["weather"]["temperature"], "<", 20)
        print(f"Temperature check result: {temp_check.result} - {temp_check.details}")
        assert temp_check.result is True
        
        # Route based on conditions
        print("\nStep 3: Route based on conditions")
        decision = {"weather": "cold_rain"}
        routes = {
            "cold_rain": "send_warning",
            "warm_rain": "send_info",
            "no_rain": "normal_update"
        }
        print(f"Decision: {decision}")
        print(f"Available routes: {routes}")
        
        router_result = conditional_router(decision, routes, decision_path="weather")
        print(f"Routing result: {router_result.routed_to} (condition: {router_result.matched_condition})")
        assert router_result.routed_to == "send_warning"

    def test_numeric_threshold_workflow(self):
        """Test numeric threshold-based workflow."""
        # Check various values against thresholds
        thresholds = {"critical": 90, "warning": 70, "normal": 30}
        
        # Critical value
        result1 = threshold_check(95, thresholds)
        assert result1.routed_to == "critical"
        
        # Warning value
        result2 = threshold_check(75, thresholds)
        assert result2.routed_to == "warning"
        
        # Normal value
        result3 = threshold_check(35, thresholds)
        assert result3.routed_to == "normal"
        
        # Below all thresholds
        result4 = threshold_check(10, thresholds, default_result="low")
        assert result4.routed_to == "low"


class TestAgenticConditionalWorkflows:
    """Test agent-based workflows using decision tools."""
    
    @pytest.mark.asyncio
    async def test_agent_weather_alert_workflow(self):
        """Test agent generating workflow that uses decision tools for weather alerts."""
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
        
        # Load environment
        env_path = Path(__file__).parent.parent / "creds.env"
        load_dotenv(env_path)
        
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
        
        print("\n=== Testing Agent-Generated Conditional Weather Workflow ===")
        
        # Create tool catalog including decision tools
        decision_tool_catalog = {
            "weather_api": {
                "name": "weather_api",
                "description": "Get weather information for a location",
                "parameters": {"location": "string", "units": "string"},
                "returns": ["weather_data", "status"]
            },
            "json_evaluator": {
                "name": "json_evaluator",
                "description": "Evaluate JSON data against expressions for decision making",
                "parameters": {"data": "dict", "expression": "string"},
                "returns": ["result", "details", "confidence"]
            },
            "number_compare": {
                "name": "number_compare", 
                "description": "Compare numbers using operators (>, <, ==, etc.)",
                "parameters": {"value": "number", "operator": "string", "threshold": "number"},
                "returns": ["result", "details", "confidence"]
            },
            "conditional_router": {
                "name": "conditional_router",
                "description": "Route to different paths based on structured decisions",
                "parameters": {"decision": "dict", "routes": "dict", "decision_path": "string"},
                "returns": ["routed_to", "route_data", "matched_condition"]
            },
            "send_alert": {
                "name": "send_alert",
                "description": "Send emergency weather alert",
                "parameters": {"message": "string", "severity": "string", "recipients": "list"},
                "returns": ["sent", "alert_id"]
            },
            "send_notification": {
                "name": "send_notification", 
                "description": "Send normal weather notification",
                "parameters": {"message": "string", "recipients": "list"},
                "returns": ["sent", "notification_id"]
            }
        }
        
        print(f"Tool catalog includes {len(decision_tool_catalog)} tools including decision tools")
        
        # Create workflow planner
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=True
        )
        
        # Request a complex conditional workflow
        complex_query = """
        Create a smart weather alert system that:
        1. Gets weather data for a city
        2. Checks if temperature is below 0°C (freezing)
        3. Checks if the condition contains 'storm' or 'severe'
        4. If either condition is true, send an emergency alert
        5. Otherwise, send a normal notification
        
        Use the decision tools (json_evaluator, number_compare, conditional_router) to implement the conditional logic properly.
        Do NOT use string conditions in edges - use explicit decision nodes instead.
        """
        
        print(f"\nComplex Query: {complex_query}")
        print("\nGenerating workflow with decision tools...")
        
        # Generate workflow
        workflow = await planner.generate_workflow(
            query=complex_query,
            tool_catalog=decision_tool_catalog
        )
        
        print("\n=== GENERATED CONDITIONAL WORKFLOW ===")
        print(f"Title: {workflow.title}")
        print(f"Description: {workflow.description}")
        
        print(f"\nNodes ({len(workflow.nodes)}):")
        decision_nodes = []
        for i, node in enumerate(workflow.nodes, 1):
            print(f"{i}. {node.id} ({node.type}): {node.label}")
            if node.data.tool_name:
                print(f"   Tool: {node.data.tool_name}")
            if node.data.config:
                print(f"   Config: {node.data.config}")
            if node.type == "decision" or (node.data.tool_name and node.data.tool_name in ["json_evaluator", "number_compare", "conditional_router"]):
                decision_nodes.append(node)
                print("   ✓ DECISION NODE DETECTED")
        
        print(f"\nEdges ({len(workflow.edges)}):")
        for edge in workflow.edges:
            condition_str = f" [condition: {edge.data.condition}]" if edge.data and edge.data.condition else ""
            print(f"  {edge.source} -> {edge.target}{condition_str}")
        
        # Validate the workflow uses decision tools properly
        print("\n=== VALIDATION ===")
        assert len(decision_nodes) > 0, "Workflow should include decision nodes"
        print(f"✓ Found {len(decision_nodes)} decision nodes")
        
        # Check for decision tool usage
        decision_tool_names = ["json_evaluator", "number_compare", "conditional_router"]
        used_decision_tools = [node.data.tool_name for node in workflow.nodes if node.data.tool_name in decision_tool_names]
        assert len(used_decision_tools) > 0, f"Workflow should use decision tools, found: {used_decision_tools}"
        print(f"✓ Uses decision tools: {used_decision_tools}")
        
        # Validate structure
        issues = workflow.validate_structure()
        assert len(issues) == 0, f"Generated workflow has structural issues: {issues}"
        print("✓ Workflow structure is valid")
        
        # Check that the workflow includes alert/notification routing
        node_tools = [node.data.tool_name for node in workflow.nodes if node.data.tool_name]
        has_alert_tools = any(tool in ["send_alert", "send_notification"] for tool in node_tools)
        assert has_alert_tools, "Workflow should include alert or notification tools"
        print("✓ Includes alert/notification tools")
        
        print("\n✅ AGENT SUCCESSFULLY GENERATED CONDITIONAL WORKFLOW USING DECISION TOOLS")
        
    def test_decision_tools_integration_direct(self):
        """Test decision tools integration directly to verify they work together."""
        print("\n=== Testing Decision Tools Integration Directly ===")
        
        # Simulate weather data processing pipeline
        weather_response = {
            "location": "Chicago",
            "current": {
                "temperature": -5,
                "condition": "severe storm",
                "humidity": 85,
                "wind_speed": 45
            },
            "alerts": {
                "active": True,
                "count": 2
            }
        }
        
        print(f"Weather data: {weather_response}")
        
        # Step 1: Check if temperature is freezing
        print("\nStep 1: Check freezing temperature")
        freezing_check = number_compare(
            weather_response["current"]["temperature"], 
            "<", 
            0
        )
        print(f"Freezing check: {freezing_check.result} - {freezing_check.details}")
        
        # Step 2: Check for severe weather conditions
        print("\nStep 2: Check for severe weather")
        severe_check = string_contains(
            weather_response["current"]["condition"],
            "severe",
            case_sensitive=False
        )
        print(f"Severe weather check: {severe_check.result} - {severe_check.details}")
        
        # Step 3: Create decision based on both conditions
        emergency_decision = {
            "alert_level": "emergency" if (freezing_check.result or severe_check.result) else "normal",
            "temperature_critical": freezing_check.result,
            "weather_severe": severe_check.result,
            "conditions": {
                "freezing": freezing_check.result,
                "severe": severe_check.result
            }
        }
        
        print(f"\nStep 3: Emergency decision: {emergency_decision}")
        
        # Step 4: Route based on alert level
        print("\nStep 4: Route notification")
        routes = {
            "emergency": "send_emergency_alert",
            "normal": "send_normal_notification"
        }
        
        routing_result = conditional_router(
            emergency_decision,
            routes,
            decision_path="alert_level"
        )
        
        print(f"Routing result: {routing_result.routed_to}")
        print(f"Route data: {routing_result.route_data}")
        print(f"Matched condition: {routing_result.matched_condition}")
        
        # Validate the decision logic worked correctly
        assert freezing_check.result is True, "Should detect freezing temperature"
        assert severe_check.result is True, "Should detect severe weather"
        assert emergency_decision["alert_level"] == "emergency", "Should escalate to emergency"
        assert routing_result.routed_to == "send_emergency_alert", "Should route to emergency alert"
        
        print("\n✅ DECISION TOOLS INTEGRATION SUCCESSFUL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])