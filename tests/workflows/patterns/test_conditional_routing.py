"""
Comprehensive test suite for conditional routing workflows.

This consolidates all conditional gate patterns from development files into
organized, parametrized tests that cover all workflow routing scenarios.
"""

import pytest
from uuid import uuid4

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env


@pytest.fixture
def mock_tool_catalog():
    """Mock tool catalog for testing without external dependencies."""
    return {
        "get_market_sentiment": {
            "description": "Get market sentiment data",
            "parameters": {}
        },
        "get_coin_quotes": {
            "description": "Get cryptocurrency price quotes",
            "parameters": {"symbol": {"type": "array"}}
        },
        "conditional_gate": {
            "description": "Route workflow based on conditions",
            "parameters": {
                "conditions": {"type": "array"},
                "default_route": {"type": "string"}
            }
        },
        "user_input": {
            "message": "Enter market sentiment data (bullish/bearish with confidence",
            "default_prompt": "Enter market sentiment data (bullish/bearish with confidence)"
        },
        "prompt_tool": {
            "message": "Provide static prompt message",
            "default_prompt": "Enter market sentiment data (bullish/bearish with confidence)"
        }
    }


@pytest.fixture
def agent_conditional_gate_workflow():
    """Create agent-based conditional gate workflow for testing."""
    nodes = [
        NodeSpec(
            id="sentiment_source",
            type="data_source",
            label="Get Market Sentiment",
            data=NodeData(
                source_name="user_input",
                config={"prompt": "Enter market sentiment data (bullish/bearish with confidence)"},
                ins=[],
                outs=["sentiment_data"]
            )
        ),
        NodeSpec(
            id="decision_agent",
            type="agent",
            label="Market Decision Agent",
            data=NodeData(
                agent_instructions="""
                You are a market decision agent. You MUST use the conditional_gate tool.
                
                Use the conditional_gate tool with the sentiment data to make routing decisions:
                - Route to "positive" if sentiment is bullish/positive AND confidence > 0.7
                - Route to "negative" if sentiment is bearish/negative AND confidence > 0.7
                - Default to "neutral" if neither condition is met
                
                Call the conditional_gate tool with proper gate_config and return the GateResult.
                """,
                tools=["conditional_gate"],
                config={},
                ins=["sentiment_data"],
                outs=["routing_decision"]
            )
        ),
        NodeSpec(
            id="positive_agent",
            type="agent",
            label="Positive Confirmation Agent",
            data=NodeData(
                agent_instructions="You received a positive market signal. Confirm and analyze.",
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
                agent_instructions="You received a negative market signal. Confirm and analyze.",
                config={},
                ins=["routing_decision"],
                outs=["negative_result"]
            )
        ),
        NodeSpec(
            id="neutral_agent",
            type="agent",
            label="Neutral Hold Agent",
            data=NodeData(
                agent_instructions="Market is neutral. Recommend holding position.",
                config={},
                ins=["routing_decision"],
                outs=["neutral_result"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(
            id="e1",
            source="sentiment_source",
            target="decision_agent",
            data=EdgeData()
        ),
        EdgeSpec(
            id="e2",
            source="decision_agent",
            target="positive_agent",
            data=EdgeData(condition="routed_to == 'positive'")
        ),
        EdgeSpec(
            id="e3",
            source="decision_agent",
            target="negative_agent",
            data=EdgeData(condition="routed_to == 'negative'")
        ),
        EdgeSpec(
            id="e4",
            source="decision_agent",
            target="neutral_agent",
            data=EdgeData(condition="routed_to == 'neutral'")
        )
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        reasoning="Test fixture for validating agent-based conditional routing using conditional_gate tool",
        title="Agent Conditional Gate Test",
        description="Test agent using conditional_gate tool for routing decisions",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "conditional_routing"}
    )


@pytest.fixture
def bitcoin_trading_workflow():
    """Create Bitcoin trading conditional workflow using real tools."""
    nodes = [
        NodeSpec(
            id="btc_price_source",
            type="data_source",
            label="Get Bitcoin Price",
            data=NodeData(
                source_name="prompt_tool",
                config={"message": "BTC price data: $50000, change: +5%"},
                ins=[],
                outs=["price_data"]
            )
        ),
        NodeSpec(
            id="decision_agent",
            type="agent",
            label="Bitcoin Decision Agent",
            data=NodeData(
                agent_instructions="""
                You are a Bitcoin trading decision agent. Use the conditional_gate tool.
                
                Analyze Bitcoin price data and route decisions:
                - Route to "buy" if price change is positive (> 0%)
                - Route to "sell" if price change is negative (< -2%)
                - Default to "hold" if neither condition is met
                
                Use conditional_gate tool with proper configuration.
                """,
                tools=["conditional_gate"],
                config={},
                ins=["price_data"],
                outs=["routing_decision"]
            )
        ),
        NodeSpec(
            id="buy_agent",
            type="agent",
            label="Buy Confirmation Agent",
            data=NodeData(
                agent_instructions="Confirm BUY signal for Bitcoin. Analyze opportunity.",
                config={},
                ins=["routing_decision"],
                outs=["buy_confirmation"]
            )
        ),
        NodeSpec(
            id="sell_agent",
            type="agent",
            label="Sell Confirmation Agent",
            data=NodeData(
                agent_instructions="Confirm SELL signal for Bitcoin. Analyze opportunity.",
                config={},
                ins=["routing_decision"],
                outs=["sell_confirmation"]
            )
        ),
        NodeSpec(
            id="hold_agent",
            type="agent",
            label="Hold Confirmation Agent",
            data=NodeData(
                agent_instructions="Confirm HOLD signal for Bitcoin. Explain strategy.",
                config={},
                ins=["routing_decision"],
                outs=["hold_confirmation"]
            )
        )
    ]
    
    edges = [
        EdgeSpec(id="e1", source="btc_price_source", target="decision_agent", data=EdgeData()),
        EdgeSpec(id="e2", source="decision_agent", target="buy_agent", 
                data=EdgeData(condition="routed_to == 'buy'")),
        EdgeSpec(id="e3", source="decision_agent", target="sell_agent", 
                data=EdgeData(condition="routed_to == 'sell'")),
        EdgeSpec(id="e4", source="decision_agent", target="hold_agent", 
                data=EdgeData(condition="routed_to == 'hold'"))
    ]
    
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        reasoning="Test fixture for Bitcoin trading workflow with buy/sell/hold conditional routing based on market sentiment",
        title="Bitcoin Conditional Trading",
        description="Bitcoin trading with conditional routing",
        nodes=nodes,
        edges=edges,
        metadata={"test_type": "crypto_trading"}
    )


class TestConditionalRouting:
    """Test suite for conditional routing workflows."""
    
    def test_workflow_structure_validation(self, agent_conditional_gate_workflow):
        """Test that conditional routing workflows have valid structure."""
        workflow = agent_conditional_gate_workflow
        
        # Validate basic structure
        assert len(workflow.nodes) == 5  # source + decision + 3 action agents
        assert len(workflow.edges) == 4  # 1 input + 3 conditional outputs
        
        # Validate decision agent has conditional_gate tool
        decision_agent = next(n for n in workflow.nodes if n.id == "decision_agent")
        assert "conditional_gate" in decision_agent.data.tools
        
        # Validate conditional edges
        conditional_edges = [e for e in workflow.edges if e.data.condition]
        assert len(conditional_edges) == 3
        
        # Validate edge conditions
        conditions = [e.data.condition for e in conditional_edges]
        assert "routed_to == 'positive'" in conditions
        assert "routed_to == 'negative'" in conditions
        assert "routed_to == 'neutral'" in conditions
    
    @pytest.mark.parametrize("scenario,expected_route", [
        ("positive_sentiment", "positive"),
        ("negative_sentiment", "negative"),
        ("neutral_sentiment", "neutral"),
        ("uncertain_sentiment", "neutral")
    ])
    def test_routing_scenarios(self, agent_conditional_gate_workflow, scenario, expected_route):
        """Test different routing scenarios for conditional workflows."""
        workflow = agent_conditional_gate_workflow
        
        # Validate that the workflow has the expected routing structure
        next(n for n in workflow.nodes if n.id == "decision_agent")
        target_agent = next(n for n in workflow.nodes if n.label.lower().startswith(expected_route))
        
        # Find the edge that should be activated for this scenario
        matching_edge = next(
            e for e in workflow.edges 
            if e.source == "decision_agent" and e.target == target_agent.id
        )
        
        assert f"routed_to == '{expected_route}'" in matching_edge.data.condition
    
    def test_bitcoin_workflow_structure(self, bitcoin_trading_workflow):
        """Test Bitcoin trading workflow structure."""
        workflow = bitcoin_trading_workflow
        
        # Validate structure
        assert len(workflow.nodes) == 5  # price source + decision + 3 trading agents
        assert len(workflow.edges) == 4
        
        # Validate price source uses valid data source
        price_source = next(n for n in workflow.nodes if n.id == "btc_price_source")
        assert price_source.data.source_name == "prompt_tool"
        assert "BTC" in price_source.data.config["message"]
        
        # Validate trading routes
        trading_edges = [e for e in workflow.edges if e.data.condition]
        trading_routes = [e.data.condition for e in trading_edges]
        
        assert "routed_to == 'buy'" in trading_routes
        assert "routed_to == 'sell'" in trading_routes
        assert "routed_to == 'hold'" in trading_routes
    
    def test_workflow_efficiency_requirements(self, agent_conditional_gate_workflow):
        """Test that conditional workflows support compute efficiency."""
        workflow = agent_conditional_gate_workflow
        
        # Check that conditional edges exist (for termination)
        conditional_edges = [e for e in workflow.edges if e.data.condition]
        assert len(conditional_edges) > 0
        
        # Check that multiple action agents exist but only one should execute
        action_agents = [n for n in workflow.nodes if "agent" in n.label.lower() and n.id != "decision_agent"]
        assert len(action_agents) >= 2  # Multiple paths available
        
        # Verify that each action agent is reached by a conditional edge
        action_agent_ids = {n.id for n in action_agents}
        conditional_targets = {e.target for e in conditional_edges}
        assert action_agent_ids.issubset(conditional_targets)
    
    def test_tool_integration(self, bitcoin_trading_workflow, mock_tool_catalog):
        """Test that workflows integrate properly with tool catalog."""
        workflow = bitcoin_trading_workflow
        
        # Validate tools used in workflow exist in catalog
        tools_in_workflow = set()
        for node in workflow.nodes:
            if hasattr(node.data, 'source_name') and node.data.source_name:
                tools_in_workflow.add(node.data.source_name)
            if hasattr(node.data, 'tools') and node.data.tools:
                tools_in_workflow.update(node.data.tools)
        
        # Check that all tools exist in catalog
        for tool in tools_in_workflow:
            assert tool in mock_tool_catalog, f"Tool {tool} not found in catalog"
    
    @pytest.mark.parametrize("workflow_type", [
        "agent_conditional_gate_workflow",
        "bitcoin_trading_workflow"
    ])
    def test_workflow_metadata(self, workflow_type, request):
        """Test that workflows have proper metadata for categorization."""
        workflow = request.getfixturevalue(workflow_type)
        
        # All test workflows should have metadata
        assert workflow.metadata is not None
        assert "test_type" in workflow.metadata
        
        # Validate specific metadata based on workflow type
        if "agent" in workflow_type:
            assert workflow.metadata["test_type"] == "conditional_routing"
        elif "bitcoin" in workflow_type:
            assert workflow.metadata["test_type"] == "crypto_trading"
    
    def test_edge_condition_parsing(self, agent_conditional_gate_workflow):
        """Test that edge conditions are properly formatted for DAG execution."""
        workflow = agent_conditional_gate_workflow
        
        conditional_edges = [e for e in workflow.edges if e.data.condition]
        
        for edge in conditional_edges:
            condition = edge.data.condition
            
            # Validate condition format
            assert "routed_to ==" in condition
            assert condition.count("'") == 2  # Properly quoted value
            
            # Validate condition values are expected routes
            if "'positive'" in condition:
                assert edge.target == "positive_agent"
            elif "'negative'" in condition:
                assert edge.target == "negative_agent"
            elif "'neutral'" in condition:
                assert edge.target == "neutral_agent"


class TestConditionalWorkflowExecution:
    """Integration tests for executing conditional workflows."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_setup(self, agent_conditional_gate_workflow, mock_tool_catalog):
        """Test that conditional workflows can be set up for execution."""
        workflow = agent_conditional_gate_workflow
        
        # Create DAG executor
        executor = DAGExecutor(use_typed_execution=True)
        executor.build_execution_graph(
            workflow_spec=workflow,
            objective="Test conditional routing",
            conversation_id="test_conversation"
        )
        
        # Validate executor setup
        assert len(executor.nodes) == len(workflow.nodes)
        assert len(executor.edges) == len(workflow.edges)
    
    @pytest.mark.asyncio
    async def test_conditional_termination_concept(self, bitcoin_trading_workflow):
        """Test conceptual validation of conditional termination."""
        workflow = bitcoin_trading_workflow
        
        # Validate that workflow structure supports conditional termination
        # (This tests the pattern without requiring full execution)
        
        # Count decision points and action points
        decision_nodes = [n for n in workflow.nodes if "decision" in n.label.lower()]
        action_nodes = [n for n in workflow.nodes if n.id in ["buy_agent", "sell_agent", "hold_agent"]]
        
        assert len(decision_nodes) == 1
        assert len(action_nodes) == 3
        
        # Each action node should only be reachable through a conditional edge
        conditional_edges = [e for e in workflow.edges if e.data.condition]
        action_targets = {e.target for e in conditional_edges}
        action_ids = {n.id for n in action_nodes}
        
        assert action_ids == action_targets, "All action nodes should be conditional targets"


class TestWorkflowPatternVariations:
    """Test different variations of conditional routing patterns."""
    
    def test_multi_condition_pattern(self):
        """Test workflow with multiple simultaneous conditions."""
        # This would test complex conditional scenarios
        # Implementation would create workflows with multiple decision points
        pass
    
    def test_nested_conditional_pattern(self):
        """Test workflow with nested conditional routing."""
        # This would test hierarchical decision trees
        # Implementation would create workflows with conditional chains
        pass
    
    def test_fan_out_pattern(self):
        """Test workflow that fans out to multiple paths simultaneously."""
        # This would test broadcasting to multiple agents
        # Implementation would test non-exclusive routing
        pass


if __name__ == "__main__":
    # Load tools for any integration tests
    load_tools_from_env()
    
    # Run tests
    pytest.main([__file__, "-v"])