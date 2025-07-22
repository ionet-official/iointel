"""
Test suite for conditional DAG termination functionality.

These tests validate that decision nodes can gate downstream execution,
preventing wasteful compute when conditions aren't met.
"""

import random
import pytest
from uuid import uuid4

from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


class TestConditionalDAGTermination:
    """Test conditional execution gating in DAG workflows."""
    
    def create_test_workflow_spec(self) -> WorkflowSpec:
        """
        Create a test workflow that mimics the Bitcoin trading scenario:
        
        data_input -> decision_node -> buy_agent
                                  -> sell_agent
                                  -> no_action
        """
        nodes = [
            NodeSpec(
                id="data_input",
                type="tool",
                label="Data Input",
                data=NodeData(
                    tool_name="mock_data_input",
                    config={"data_type": "market_data"},
                    ins=[],
                    outs=["market_data"]
                )
            ),
            NodeSpec(
                id="decision_node",
                type="decision",
                label="Trading Decision",
                data=NodeData(
                    tool_name="conditional_router",
                    config={
                        "routes": {
                            "buy": "buy_path",
                            "sell": "sell_path", 
                            "no_action": "no_action_path"
                        },
                        "default_route": "no_action_path",
                        "decision_path": "action"
                    },
                    ins=["market_data"],
                    outs=["decision_result"]
                )
            ),
            NodeSpec(
                id="buy_agent",
                type="agent",
                label="Buy Agent",
                data=NodeData(
                    agent_instructions="Execute buy order",
                    ins=["decision_result"],
                    outs=["buy_result"]
                )
            ),
            NodeSpec(
                id="sell_agent", 
                type="agent",
                label="Sell Agent",
                data=NodeData(
                    agent_instructions="Execute sell order",
                    ins=["decision_result"],
                    outs=["sell_result"]
                )
            ),
            NodeSpec(
                id="no_action_agent",
                type="agent",
                label="No Action Agent", 
                data=NodeData(
                    agent_instructions="Log no action taken",
                    ins=["decision_result"],
                    outs=["no_action_result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="data_input",
                target="decision_node",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge_2",
                source="decision_node",
                target="buy_agent", 
                data=EdgeData(condition="routed_to == 'buy_path'")
            ),
            EdgeSpec(
                id="edge_3",
                source="decision_node",
                target="sell_agent",
                data=EdgeData(condition="routed_to == 'sell_path'")
            ),
            EdgeSpec(
                id="edge_4",
                source="decision_node",
                target="no_action_agent",
                data=EdgeData(condition="routed_to == 'no_action_path'")
            )
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Test Conditional Workflow",
            description="Test workflow for conditional DAG termination",
            nodes=nodes,
            edges=edges
        )
    
    async def mock_data_input_executor(self, task_metadata, objective, agents, execution_metadata):
        """Mock data input that returns market data."""
        todays_change = random.uniform(-7.0, 7.0)
        return {
            "result": {
                "price": 100.0,
                "change_percent": todays_change,  # Random value between -7% and 7%
                "action": "sell" if todays_change > 0.5 else ("buy" if todays_change < -0.5 else "no_action")  # This will route to no_action_path
            }
        }
    
    async def mock_decision_executor(self, task_metadata, objective, agents, execution_metadata):
        """Mock decision executor that uses conditional_router logic."""
        config = task_metadata.get("config", {})
        routes = config.get("routes", {})
        default_route = config.get("default_route", "no_action_path")
        
        # Simulate decision logic - in real scenario this would come from previous node
        decision_data = {"action": "no_action"}  # Default to no action
        
        action = decision_data.get("action", "no_action")
        routed_to = routes.get(action, default_route)
        
        return {
            "result": {
                "routed_to": routed_to,
                "route_data": {"decision_value": action},
                "matched_condition": f"action == {action}"
            }
        }
    
    async def mock_agent_executor(self, task_metadata, objective, agents, execution_metadata):
        """Mock agent executor that simulates expensive computation."""
        agent_name = task_metadata.get("agent_instructions", "unknown")
        return {
            "result": f"Agent executed: {agent_name}"
        }
    
    async def mock_sentiment_executor(self, task_metadata, objective, agents, execution_metadata):
        """Mock sentiment data provider."""
        return {
            "result": {
                "sentiment": "bullish",
                "confidence": 0.85,
                "volume": 5000
            }
        }


    @pytest.fixture
    def setup_executors(self):
        """Register mock executors for testing."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        
        # Store original executors
        original_executors = TASK_EXECUTOR_REGISTRY.copy()
        
        # Register mock executors
        TASK_EXECUTOR_REGISTRY["tool"] = self.mock_data_input_executor
        TASK_EXECUTOR_REGISTRY["decision"] = self.mock_decision_executor
        TASK_EXECUTOR_REGISTRY["agent"] = self.mock_agent_executor
        
        yield
        
        # Restore original executors
        TASK_EXECUTOR_REGISTRY.clear()
        TASK_EXECUTOR_REGISTRY.update(original_executors)
    
    @pytest.mark.asyncio
    async def test_full_termination_no_action(self, setup_executors):
        """Test that all downstream agents are skipped when decision routes to no_action."""
        spec = self.create_test_workflow_spec()
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test no action scenario"
        )
        
        # Execute workflow
        initial_state = WorkflowState(
            conversation_id="test_no_action",
            initial_text="Test input",
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        
        # Verify execution results
        assert "data_input" in final_state.results
        assert "decision_node" in final_state.results
        assert "no_action_agent" in final_state.results
        
        # Verify buy and sell agents were skipped
        assert "buy_agent" in final_state.results
        assert "sell_agent" in final_state.results
        assert final_state.results["buy_agent"]["status"] == "skipped"
        assert final_state.results["sell_agent"]["status"] == "skipped"
        
        # Verify execution statistics
        stats = executor.get_execution_statistics()
        assert stats["total_nodes"] == 5
        assert stats["executed_nodes"] == 3  # data_input, decision_node, no_action_agent
        assert stats["skipped_nodes"] == 2   # buy_agent, sell_agent
        assert "buy_agent" in stats["skipped_node_ids"]
        assert "sell_agent" in stats["skipped_node_ids"]
        assert "60.0%" in stats["execution_efficiency"]  # 40% compute saved
    
    async def mock_decision_executor_buy(self, task_metadata, objective, agents, execution_metadata):
        """Mock decision executor that routes to buy."""
        config = task_metadata.get("config", {})
        routes = config.get("routes", {})
        
        return {
            "result": {
                "routed_to": routes.get("buy", "buy_path"),
                "route_data": {"decision_value": "buy"},
                "matched_condition": "action == buy"
            }
        }
    
    @pytest.mark.asyncio
    async def test_partial_execution_buy_path(self, setup_executors):
        """Test that only buy agent executes when decision routes to buy."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        
        # Override decision executor to route to buy
        TASK_EXECUTOR_REGISTRY["decision"] = self.mock_decision_executor_buy
        
        spec = self.create_test_workflow_spec()
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test buy scenario"
        )
        
        # Execute workflow
        initial_state = WorkflowState(
            conversation_id="test_buy",
            initial_text="Test input",
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        
        # Verify execution results
        assert "data_input" in final_state.results
        assert "decision_node" in final_state.results
        assert "buy_agent" in final_state.results
        
        # Verify buy agent executed (not skipped)
        buy_result = final_state.results["buy_agent"]
        if isinstance(buy_result, dict):
            assert buy_result.get("status") != "skipped"
        else:
            # If it's a string, it means the agent executed successfully
            assert isinstance(buy_result, str)
        
        # Verify sell and no_action agents were skipped
        assert final_state.results["sell_agent"]["status"] == "skipped"
        assert final_state.results["no_action_agent"]["status"] == "skipped"
        
        # Verify execution statistics
        stats = executor.get_execution_statistics()
        assert stats["total_nodes"] == 5
        assert stats["executed_nodes"] == 3  # data_input, decision_node, buy_agent
        assert stats["skipped_nodes"] == 2   # sell_agent, no_action_agent
        assert "sell_agent" in stats["skipped_node_ids"]
        assert "no_action_agent" in stats["skipped_node_ids"]
    
    async def mock_decision_executor_all_false(self, task_metadata, objective, agents, execution_metadata):
        """Mock decision executor that routes to non-existent path (all false)."""
        return {
            "result": {
                "routed_to": "non_existent_path",
                "route_data": {"decision_value": "unknown"},
                "matched_condition": None
            }
        }
    
    @pytest.mark.asyncio
    async def test_full_termination_all_false(self, setup_executors):
        """Test that all downstream agents are skipped when decision routes to non-existent path."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        
        # Override decision executor to route to non-existent path
        TASK_EXECUTOR_REGISTRY["decision"] = self.mock_decision_executor_all_false
        
        spec = self.create_test_workflow_spec()
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test all false scenario"
        )
        
        # Execute workflow
        initial_state = WorkflowState(
            conversation_id="test_all_false",
            initial_text="Test input",
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        
        # Verify execution results
        assert "data_input" in final_state.results
        assert "decision_node" in final_state.results
        
        # Verify ALL downstream agents were skipped
        assert final_state.results["buy_agent"]["status"] == "skipped"
        assert final_state.results["sell_agent"]["status"] == "skipped"
        assert final_state.results["no_action_agent"]["status"] == "skipped"
        
        # Verify execution statistics show maximum compute savings
        stats = executor.get_execution_statistics()
        assert stats["total_nodes"] == 5
        assert stats["executed_nodes"] == 2  # Only data_input, decision_node
        assert stats["skipped_nodes"] == 3   # All downstream agents
        assert len(stats["skipped_node_ids"]) == 3
        assert "40.0%" in stats["execution_efficiency"]  # 60% compute saved
    
    @pytest.mark.asyncio
    async def test_boolean_decision_gating(self, setup_executors):
        """Test that boolean decision results also gate execution."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        
        async def mock_boolean_decision(task_metadata, objective, agents, execution_metadata):
            """Mock decision that returns boolean result."""
            return {
                "result": {
                    "result": False,  # Boolean false should gate downstream
                    "details": "Condition not met"
                }
            }
        
        TASK_EXECUTOR_REGISTRY["decision"] = mock_boolean_decision
        
        spec = self.create_test_workflow_spec()
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test boolean gating"
        )
        
        # Execute workflow
        initial_state = WorkflowState(
            conversation_id="test_boolean",
            initial_text="Test input",
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        
        # Verify ALL downstream agents were skipped due to boolean false
        assert final_state.results["buy_agent"]["status"] == "skipped"
        assert final_state.results["sell_agent"]["status"] == "skipped"
        assert final_state.results["no_action_agent"]["status"] == "skipped"
        
        # Verify execution statistics
        stats = executor.get_execution_statistics()
        assert stats["skipped_nodes"] == 3
        assert stats["executed_nodes"] == 2

    @pytest.mark.asyncio
    async def test_agent_conditional_gate_workflow(self, setup_executors):
        """Test agent using conditional_gate tool in workflow spec for routing decisions."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        from iointel.src.web.workflow_server import web_tool_executor, web_agent_executor
        
        # Register proper executors for this test
        TASK_EXECUTOR_REGISTRY["tool"] = mock_sentiment_tool_executor
        TASK_EXECUTOR_REGISTRY["agent"] = web_agent_executor
        TASK_EXECUTOR_REGISTRY["decision"] = web_tool_executor
        
        # Create workflow where agent uses conditional_gate tool
        # Agents will be auto-created from WorkflowSpec node data
        
        nodes = [
            NodeSpec(
                id="sentiment_source",
                type="tool",
                label="Mock Market Sentiment",
                data=NodeData(
                    tool_name="mock_sentiment_provider",
                    config={},
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
                    - Default to "none" if neither condition is met
                    
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
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Agent Conditional Gate Test",
            description="Test agent using conditional_gate tool for routing",
            nodes=nodes,
            edges=edges
        )
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test agent conditional gating"
        )
        
        # Execute workflow
        initial_state = WorkflowState(
            conversation_id="test_agent_gate",
            initial_text="Test agent conditional gating",
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        
        # Verify basic execution
        assert "sentiment_source" in final_state.results
        assert "decision_agent" in final_state.results
        
        # Get decision agent result
        decision_result = final_state.results["decision_agent"]
        
        # Verify agent used conditional_gate tool
        if isinstance(decision_result, dict) and "tool_usage_results" in decision_result:
            tool_usage = decision_result["tool_usage_results"]
            assert len(tool_usage) > 0, "Agent should have used conditional_gate tool"
            assert any(usage.tool_name == "conditional_gate" for usage in tool_usage), \
                "Agent should have called conditional_gate tool"
            
            # Check the gate result
            gate_usage = next(usage for usage in tool_usage if usage.tool_name == "conditional_gate")
            gate_result = gate_usage.tool_result
            
            # Verify GateResult structure
            assert hasattr(gate_result, 'routed_to'), "GateResult should have routed_to field"
            assert hasattr(gate_result, 'action'), "GateResult should have action field"
            assert hasattr(gate_result, 'decision_reason'), "GateResult should have decision_reason field"
            
            # Verify routing logic based on routed_to
            if gate_result.routed_to == "positive":
                # Positive route: positive_agent should execute, negative_agent should be skipped
                assert "positive_agent" in final_state.results
                assert isinstance(final_state.results["positive_agent"], dict)
                assert final_state.results["positive_agent"].get("status") != "skipped"
                
                # Negative agent should be skipped
                if "negative_agent" in final_state.results:
                    assert final_state.results["negative_agent"].get("status") == "skipped"
            
            elif gate_result.routed_to == "negative":
                # Negative route: negative_agent should execute, positive_agent should be skipped
                assert "negative_agent" in final_state.results
                assert isinstance(final_state.results["negative_agent"], dict)
                assert final_state.results["negative_agent"].get("status") != "skipped"
                
                # Positive agent should be skipped
                if "positive_agent" in final_state.results:
                    assert final_state.results["positive_agent"].get("status") == "skipped"
            
            elif gate_result.routed_to == "none":
                # None route: both agents should be skipped (true termination)
                if "positive_agent" in final_state.results:
                    assert final_state.results["positive_agent"].get("status") == "skipped"
                if "negative_agent" in final_state.results:
                    assert final_state.results["negative_agent"].get("status") == "skipped"
        
        # Verify execution statistics show conditional gating worked
        stats = executor.get_execution_statistics()
        assert stats["total_nodes"] == 4
        
        # Should have some compute savings from conditional gating
        efficiency_parts = stats["execution_efficiency"].split("/")
        executed = int(efficiency_parts[0])
        total = int(efficiency_parts[1].split()[0])
        assert executed <= total, "Executed nodes should be <= total nodes"
        
        # If any agents were skipped, verify they're in skipped_node_ids
        if stats["skipped_nodes"] > 0:
            assert len(stats["skipped_node_ids"]) == stats["skipped_nodes"]
            skipped_agents = [node_id for node_id in stats["skipped_node_ids"] 
                            if node_id in ["positive_agent", "negative_agent"]]
            assert len(skipped_agents) > 0, "Should have skipped at least one confirmation agent"


# Standalone mock functions for task executors (no self parameter)
async def mock_sentiment_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Mock sentiment data provider."""
    return {
        "result": {
            "sentiment": "bullish",
            "confidence": 0.85,
            "volume": 5000
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])