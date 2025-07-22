"""
Comprehensive test suite for user input workflows.

This consolidates all user input and interactive workflow patterns from development
files into organized tests covering user interaction and data collection scenarios.
"""

import pytest
from uuid import uuid4

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


@pytest.fixture
def user_input_conditional_workflow():
    """Create user input conditional workflow for testing."""
    nodes = [
        NodeSpec(
            id="user_input_source",
            type="tool",
            label="Market Sentiment Input",
            data=NodeData(
                tool_name="user_input",
                config={
                    "prompt": "Enter market sentiment data (format: 'sentiment:bullish,confidence:0.8')",
                    "placeholder": "sentiment:bullish,confidence:0.8"
                },
                ins=[],
                outs=["sentiment_input"]
            )
        ),
        NodeSpec(
            id="decision_agent",
            type="agent",
            label="Market Decision Agent",
            data=NodeData(
                agent_instructions="""
                Parse user input format: "sentiment:VALUE,confidence:NUMBER"
                Use conditional_gate tool with logic:
                - Route to "positive" if sentiment is bullish/positive AND confidence > 0.7
                - Route to "negative" if sentiment is bearish/negative AND confidence > 0.7
                - Route to "none" if confidence <= 0.7 or sentiment is neutral
                """,
                tools=["conditional_gate"],
                config={},
                ins=["sentiment_input"],
                outs=["routing_decision"]
            )
        ),
        NodeSpec(
            id="positive_agent",
            type="agent",
            label="Positive Confirmation Agent",
            data=NodeData(
                agent_instructions="Confirm positive market signal and analyze opportunity.",
                config={},
                ins=["routing_decision"],
                outs=["positive_result"]
            )
        ),
        NodeSpec(
            id="negative_agent",
            type="agent",
            label="Negative Confirmation Agent",
            data=NodeData(
                agent_instructions="Confirm negative market signal and analyze opportunity.",
                config={},
                ins=["routing_decision"],
                outs=["negative_result"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="e1", source="user_input_source", target="decision_agent", data=EdgeData()),
        EdgeSpec(id="e2", source="decision_agent", target="positive_agent", 
                data=EdgeData(condition="routed_to == 'positive'")),
        EdgeSpec(id="e3", source="decision_agent", target="negative_agent", 
                data=EdgeData(condition="routed_to == 'negative'"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="User Input Conditional Gate",
        description="Interactive conditional workflow using user_input tool",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "user_input_conditional"}
    )


@pytest.fixture
def smart_input_workflow():
    """Create smart input workflow with fallback mechanisms."""
    nodes = [
        NodeSpec(
            id="smart_input",
            type="tool",
            label="Smart User Input",
            data=NodeData(
                tool_name="user_input",
                config={
                    "prompt": "Enter investment amount (USD) or 'auto' for smart allocation:",
                    "placeholder": "1000",
                    "validation_pattern": r"^(\d+(\.\d{2})?|auto)$"
                },
                ins=[],
                outs=["investment_input"]
            )
        ),
        NodeSpec(
            id="input_processor",
            type="agent",
            label="Input Processing Agent",
            data=NodeData(
                agent_instructions="""
                Process user investment input:
                - If numeric value: validate amount is reasonable (> 10, < 1000000)
                - If 'auto': calculate smart allocation based on portfolio
                - Use conditional_gate to route based on validation result
                """,
                tools=["conditional_gate"],
                config={},
                ins=["investment_input"],
                outs=["processed_input"]
            )
        ),
        NodeSpec(
            id="manual_investment_agent",
            type="agent",
            label="Manual Investment Agent",
            data=NodeData(
                agent_instructions="Process manual investment amount. Validate and execute.",
                config={},
                ins=["processed_input"],
                outs=["manual_result"]
            )
        ),
        NodeSpec(
            id="auto_allocation_agent",
            type="agent",
            label="Auto Allocation Agent",
            data=NodeData(
                agent_instructions="Calculate and execute smart portfolio allocation.",
                config={},
                ins=["processed_input"],
                outs=["auto_result"]
            )
        ),
        NodeSpec(
            id="error_handler_agent",
            type="agent",
            label="Input Error Handler",
            data=NodeData(
                agent_instructions="Handle invalid input. Provide user guidance.",
                config={},
                ins=["processed_input"],
                outs=["error_guidance"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="e1", source="smart_input", target="input_processor", data=EdgeData()),
        EdgeSpec(id="e2", source="input_processor", target="manual_investment_agent", 
                data=EdgeData(condition="routed_to == 'manual'")),
        EdgeSpec(id="e3", source="input_processor", target="auto_allocation_agent", 
                data=EdgeData(condition="routed_to == 'auto'")),
        EdgeSpec(id="e4", source="input_processor", target="error_handler_agent", 
                data=EdgeData(condition="routed_to == 'error'"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Smart Input Investment Workflow",
        description="Smart input processing with validation and fallback",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "smart_input"}
    )


@pytest.fixture
def prompt_based_workflow():
    """Create workflow using prompt_tool for dynamic prompt generation."""
    nodes = [
        NodeSpec(
            id="prompt_generator",
            type="tool",
            label="Dynamic Prompt Generator",
            data=NodeData(
                tool_name="prompt_tool",
                config={
                    "template": "Based on current market conditions: {market_data}, generate a trading question for the user.",
                    "variables": {"market_data": "crypto_volatile_session"}
                },
                ins=[],
                outs=["dynamic_prompt"]
            )
        ),
        NodeSpec(
            id="user_response",
            type="tool",
            label="User Response Input",
            data=NodeData(
                tool_name="user_input",
                config={
                    "prompt": "{dynamic_prompt}",  # Uses output from prompt generator
                    "placeholder": "Enter your trading preference"
                },
                ins=["dynamic_prompt"],
                outs=["user_trading_preference"]
            )
        ),
        NodeSpec(
            id="preference_analyzer",
            type="agent",
            label="Trading Preference Analyzer",
            data=NodeData(
                agent_instructions="""
                Analyze user trading preference and route accordingly:
                - Route to "aggressive" for high-risk preferences
                - Route to "conservative" for low-risk preferences
                - Route to "balanced" for moderate preferences
                """,
                tools=["conditional_gate"],
                config={},
                ins=["user_trading_preference"],
                outs=["preference_routing"]
            )
        ),
        NodeSpec(
            id="aggressive_strategy_agent",
            type="agent",
            label="Aggressive Strategy Agent",
            data=NodeData(
                agent_instructions="Implement aggressive trading strategy based on user preference.",
                config={},
                ins=["preference_routing"],
                outs=["aggressive_strategy"]
            )
        ),
        NodeSpec(
            id="conservative_strategy_agent",
            type="agent",
            label="Conservative Strategy Agent",
            data=NodeData(
                agent_instructions="Implement conservative trading strategy based on user preference.",
                config={},
                ins=["preference_routing"],
                outs=["conservative_strategy"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="e1", source="prompt_generator", target="user_response", data=EdgeData()),
        EdgeSpec(id="e2", source="user_response", target="preference_analyzer", data=EdgeData()),
        EdgeSpec(id="e3", source="preference_analyzer", target="aggressive_strategy_agent", 
                data=EdgeData(condition="routed_to == 'aggressive'")),
        EdgeSpec(id="e4", source="preference_analyzer", target="conservative_strategy_agent", 
                data=EdgeData(condition="routed_to == 'conservative'"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Dynamic Prompt Trading Workflow",
        description="Dynamic prompt generation with user interaction",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "prompt_based"}
    )


class TestUserInputWorkflows:
    """Test suite for user input and interactive workflows."""
    
    def test_user_input_workflow_structure(self, user_input_conditional_workflow):
        """Test basic structure of user input workflows."""
        workflow = user_input_conditional_workflow
        
        # Validate basic structure
        assert len(workflow.nodes) == 4  # input + decision + 2 action agents
        assert len(workflow.edges) == 3
        
        # Find user input node
        input_node = next(n for n in workflow.nodes if n.data.tool_name == "user_input")
        assert input_node is not None
        assert "prompt" in input_node.data.config
        assert "placeholder" in input_node.data.config
        
        # Validate decision agent has conditional_gate
        decision_agent = next(n for n in workflow.nodes if "decision" in n.label.lower())
        assert "conditional_gate" in decision_agent.data.tools
    
    def test_smart_input_workflow_validation(self, smart_input_workflow):
        """Test smart input workflow with validation and fallback."""
        workflow = smart_input_workflow
        
        # Validate structure supports multiple routing outcomes
        assert len(workflow.nodes) == 5  # input + processor + 3 handlers
        assert len(workflow.edges) == 4
        
        # Validate input validation configuration
        input_node = next(n for n in workflow.nodes if n.data.tool_name == "user_input")
        assert "validation_pattern" in input_node.data.config
        
        # Validate routing targets
        processor = next(n for n in workflow.nodes if "processor" in n.id or "processing" in n.label.lower())
        routing_edges = [e for e in workflow.edges if e.source == processor.id]
        
        expected_routes = {"manual", "auto", "error"}
        actual_routes = set()
        for edge in routing_edges:
            condition = edge.data.condition
            if "routed_to == " in condition:
                route = condition.split("'")[1]
                actual_routes.add(route)
        
        assert expected_routes.issubset(actual_routes)
    
    def test_prompt_based_workflow_chaining(self, prompt_based_workflow):
        """Test workflow with dynamic prompt generation chaining."""
        workflow = prompt_based_workflow
        
        # Validate prompt generation chain
        prompt_gen = next(n for n in workflow.nodes if n.data.tool_name == "prompt_tool")
        user_input = next(n for n in workflow.nodes if n.data.tool_name == "user_input")
        
        # Check that user_input receives prompt from generator
        connecting_edge = next(e for e in workflow.edges 
                             if e.source == prompt_gen.id and e.target == user_input.id)
        assert connecting_edge is not None
        
        # Validate user_input config references dynamic prompt
        assert "{dynamic_prompt}" in user_input.data.config["prompt"]
    
    @pytest.mark.parametrize("input_type,expected_validation", [
        ("user_input", ["prompt", "placeholder"]),
        ("smart_input", ["prompt", "placeholder", "validation_pattern"]),
        ("prompt_based", ["template", "variables"])
    ])
    def test_input_configuration_requirements(self, input_type, expected_validation, request):
        """Test that different input types have required configuration."""
        if input_type == "user_input":
            workflow = request.getfixturevalue("user_input_conditional_workflow")
            tool_name = "user_input"
        elif input_type == "smart_input":
            workflow = request.getfixturevalue("smart_input_workflow")
            tool_name = "user_input"
        elif input_type == "prompt_based":
            workflow = request.getfixturevalue("prompt_based_workflow")
            tool_name = "prompt_tool"
        
        # Find the input node
        input_node = next(n for n in workflow.nodes if n.data.tool_name == tool_name)
        
        # Validate required configuration
        for required_field in expected_validation:
            assert required_field in input_node.data.config, \
                f"Missing required field {required_field} in {input_type} config"
    
    def test_user_input_routing_scenarios(self, user_input_conditional_workflow):
        """Test different user input scenarios and expected routing."""
        workflow = user_input_conditional_workflow
        
        # Test scenarios that should be handled
        
        # Validate workflow structure supports these scenarios
        decision_agent = next(n for n in workflow.nodes if "decision" in n.label.lower())
        
        # Check that agent instructions mention the routing logic
        instructions = decision_agent.data.agent_instructions.lower()
        assert "bullish" in instructions or "positive" in instructions
        assert "bearish" in instructions or "negative" in instructions
        assert "confidence" in instructions
        assert "0.7" in instructions  # Confidence threshold
    
    def test_interactive_workflow_metadata(self, user_input_conditional_workflow, 
                                         smart_input_workflow, prompt_based_workflow):
        """Test that interactive workflows have appropriate metadata."""
        workflows = [user_input_conditional_workflow, smart_input_workflow, prompt_based_workflow]
        
        for workflow in workflows:
            assert workflow.metadata is not None
            assert "test_type" in workflow.metadata
            
            # Interactive workflows should be marked appropriately
            test_type = workflow.metadata["test_type"]
            assert any(keyword in test_type for keyword in ["input", "prompt", "interactive"])
    
    def test_workflow_user_experience_design(self, smart_input_workflow):
        """Test that workflows are designed for good user experience."""
        workflow = smart_input_workflow
        
        # Validate UX considerations
        input_node = next(n for n in workflow.nodes if n.data.tool_name == "user_input")
        
        # Should have helpful prompt text
        prompt = input_node.data.config["prompt"]
        assert len(prompt) > 20  # Descriptive prompt
        assert "(" in prompt and ")" in prompt  # Contains guidance
        
        # Should have example placeholder
        placeholder = input_node.data.config["placeholder"]
        assert len(placeholder) > 0
        
        # Should have validation pattern for user guidance
        assert "validation_pattern" in input_node.data.config
        
        # Should have error handling path
        error_handler = next(n for n in workflow.nodes if "error" in n.label.lower())
        assert "guidance" in error_handler.label.lower() or "handler" in error_handler.label.lower()
    
    def test_input_data_flow_integrity(self, user_input_conditional_workflow):
        """Test that data flows correctly through user input workflows."""
        workflow = user_input_conditional_workflow
        
        # Trace data flow from input to decision to action
        input_node = next(n for n in workflow.nodes if n.data.tool_name == "user_input")
        decision_node = next(n for n in workflow.nodes if "decision" in n.label.lower())
        
        # Input should connect to decision
        input_to_decision = next(e for e in workflow.edges 
                               if e.source == input_node.id and e.target == decision_node.id)
        assert input_to_decision is not None
        
        # Decision should have conditional outputs
        decision_outputs = [e for e in workflow.edges if e.source == decision_node.id]
        conditional_outputs = [e for e in decision_outputs if e.data.condition]
        
        assert len(conditional_outputs) >= 2  # At least positive and negative routes
        
        # Each conditional output should target a different agent
        targets = {e.target for e in conditional_outputs}
        assert len(targets) == len(conditional_outputs)  # All unique targets


class TestInteractiveWorkflowPatterns:
    """Test advanced patterns for interactive workflows."""
    
    def test_multi_step_input_pattern(self):
        """Test workflow with multiple sequential user inputs."""
        # This would test workflows that collect multiple pieces of data
        # Implementation would create workflows with multiple user_input nodes
        pass
    
    def test_conditional_input_collection(self):
        """Test workflow that conditionally collects different inputs."""
        # This would test workflows where the input collection depends on previous choices
        # Implementation would test branching input collection patterns
        pass
    
    def test_input_validation_and_retry(self):
        """Test workflow with input validation and retry mechanisms."""
        # This would test robust input handling with validation loops
        # Implementation would test error handling and re-prompting patterns
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])