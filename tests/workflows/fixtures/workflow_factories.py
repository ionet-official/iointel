"""
Factory functions for creating test workflows and shared fixtures.

This module provides reusable factories for creating workflow specifications
for testing different patterns and scenarios.
"""

from uuid import uuid4
from typing import Dict, List, Any, Optional
from datetime import datetime

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


class WorkflowFactory:
    """Factory for creating standardized test workflows."""
    
    @staticmethod
    def create_basic_conditional_workflow(
        title: str = "Basic Conditional Test Workflow",
        conditions: Optional[Dict[str, str]] = None
    ) -> WorkflowSpec:
        """Create a basic conditional workflow for testing."""
        if conditions is None:
            conditions = {
                "positive": "routed_to == 'positive'",
                "negative": "routed_to == 'negative'"
            }
        
        nodes = [
            NodeSpec(
                id="input_source",
                type="tool",
                label="Data Source",
                data=NodeData(
                    tool_name="get_market_sentiment",
                    config={},
                    ins=[],
                    outs=["data"]
                )
            ),
            NodeSpec(
                id="decision_agent",
                type="agent",
                label="Decision Agent",
                data=NodeData(
                    agent_instructions="Use conditional_gate tool to route based on data.",
                    tools=["conditional_gate"],
                    config={},
                    ins=["data"],
                    outs=list(conditions.keys())
                )
            )
        ]
        
        edges = [
            EdgeSpec(id="input_to_decision", source="input_source", target="decision_agent", data=EdgeData())
        ]
        
        # Add action nodes and edges for each condition
        for i, (route, condition) in enumerate(conditions.items()):
            action_node = NodeSpec(
                id=f"{route}_action",
                type="agent",
                label=f"{route.title()} Action Agent",
                data=NodeData(
                    agent_instructions=f"Handle {route} routing scenario.",
                    config={},
                    ins=[route],
                    outs=[f"{route}_result"]
                )
            )
            nodes.append(action_node)
            
            edge = EdgeSpec(
                id=f"decision_to_{route}",
                source="decision_agent",
                target=f"{route}_action",
                data=EdgeData(condition=condition)
            )
            edges.append(edge)
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title=title,
            description=f"Basic conditional workflow with {len(conditions)} routing paths",
            nodes=nodes,
            edges=edges,
            metadata={"test_type": "basic_conditional", "factory": "created"}
        )
    
    @staticmethod
    def create_user_input_workflow(
        prompt: str = "Enter test data:",
        placeholder: str = "test_value",
        validation_pattern: Optional[str] = None
    ) -> WorkflowSpec:
        """Create a user input workflow for testing."""
        config = {
            "prompt": prompt,
            "placeholder": placeholder
        }
        if validation_pattern:
            config["validation_pattern"] = validation_pattern
        
        nodes = [
            NodeSpec(
                id="user_input_node",
                type="tool",
                label="User Input",
                data=NodeData(
                    tool_name="user_input",
                    config=config,
                    ins=[],
                    outs=["user_data"]
                )
            ),
            NodeSpec(
                id="processing_agent",
                type="agent",
                label="Input Processing Agent",
                data=NodeData(
                    agent_instructions="Process user input and provide response.",
                    config={},
                    ins=["user_data"],
                    outs=["processed_result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(id="input_to_processing", source="user_input_node", target="processing_agent", data=EdgeData())
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="User Input Test Workflow",
            description="Test workflow for user input collection and processing",
            nodes=nodes,
            edges=edges,
            metadata={"test_type": "user_input", "factory": "created"}
        )
    
    @staticmethod
    def create_crypto_trading_workflow(
        symbol: str = "BTC",
        trading_strategies: Optional[List[str]] = None
    ) -> WorkflowSpec:
        """Create a crypto trading workflow for testing."""
        if trading_strategies is None:
            trading_strategies = ["buy", "sell", "hold"]
        
        nodes = [
            NodeSpec(
                id="price_source",
                type="tool",
                label=f"Get {symbol} Price",
                data=NodeData(
                    tool_name="get_coin_quotes",
                    config={"symbol": [symbol]},
                    ins=[],
                    outs=["price_data"]
                )
            ),
            NodeSpec(
                id="trading_decision_agent",
                type="agent",
                label="Trading Decision Agent",
                data=NodeData(
                    agent_instructions=f"""Analyze {symbol} price data and make trading decisions.
                    Use conditional_gate tool to route to: {', '.join(trading_strategies)}""",
                    tools=["conditional_gate"],
                    config={},
                    ins=["price_data"],
                    outs=trading_strategies
                )
            )
        ]
        
        edges = [
            EdgeSpec(id="price_to_decision", source="price_source", target="trading_decision_agent", data=EdgeData())
        ]
        
        # Add strategy execution nodes
        for strategy in trading_strategies:
            strategy_node = NodeSpec(
                id=f"{strategy}_execution",
                type="agent",
                label=f"{strategy.title()} Strategy Execution",
                data=NodeData(
                    agent_instructions=f"Execute {strategy} strategy for {symbol}.",
                    config={},
                    ins=[strategy],
                    outs=[f"{strategy}_result"]
                )
            )
            nodes.append(strategy_node)
            
            edge = EdgeSpec(
                id=f"decision_to_{strategy}",
                source="trading_decision_agent",
                target=f"{strategy}_execution",
                data=EdgeData(condition=f"routed_to == '{strategy}'")
            )
            edges.append(edge)
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title=f"{symbol} Trading Workflow",
            description=f"Crypto trading workflow for {symbol} with {len(trading_strategies)} strategies",
            nodes=nodes,
            edges=edges,
            metadata={"test_type": "crypto_trading", "symbol": symbol, "factory": "created"}
        )
    
    @staticmethod
    def create_multi_tool_workflow(
        tools: List[str],
        workflow_pattern: str = "sequential"
    ) -> WorkflowSpec:
        """Create workflow using multiple tools for testing tool integration."""
        nodes = []
        edges = []
        
        if workflow_pattern == "sequential":
            # Create sequential workflow where tools connect in order
            for i, tool in enumerate(tools):
                node = NodeSpec(
                    id=f"tool_{i}_{tool.replace('.', '_')}",
                    type="tool",
                    label=f"Tool: {tool}",
                    data=NodeData(
                        tool_name=tool,
                        config={},
                        ins=["input"] if i == 0 else [f"output_{i-1}"],
                        outs=[f"output_{i}"]
                    )
                )
                nodes.append(node)
                
                if i > 0:
                    edge = EdgeSpec(
                        id=f"tool_{i-1}_to_tool_{i}",
                        source=f"tool_{i-1}_{tools[i-1].replace('.', '_')}",
                        target=f"tool_{i}_{tool.replace('.', '_')}",
                        data=EdgeData()
                    )
                    edges.append(edge)
        
        elif workflow_pattern == "parallel":
            # Create parallel workflow where all tools run independently
            for i, tool in enumerate(tools):
                node = NodeSpec(
                    id=f"parallel_tool_{i}_{tool.replace('.', '_')}",
                    type="tool",
                    label=f"Parallel Tool: {tool}",
                    data=NodeData(
                        tool_name=tool,
                        config={},
                        ins=["shared_input"],
                        outs=[f"parallel_output_{i}"]
                    )
                )
                nodes.append(node)
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title=f"Multi-Tool {workflow_pattern.title()} Workflow",
            description=f"{workflow_pattern.title()} workflow using {len(tools)} tools",
            nodes=nodes,
            edges=edges,
            metadata={"test_type": "multi_tool", "pattern": workflow_pattern, "factory": "created"}
        )
    
    @staticmethod
    def create_error_handling_workflow() -> WorkflowSpec:
        """Create workflow with error handling and fallback paths."""
        nodes = [
            NodeSpec(
                id="risky_operation",
                type="tool",
                label="Risky Operation",
                data=NodeData(
                    tool_name="get_coin_quotes",  # Could fail if API is down
                    config={"symbol": ["INVALID_SYMBOL"]},
                    ins=[],
                    outs=["operation_result", "operation_error"]
                )
            ),
            NodeSpec(
                id="error_handler",
                type="agent",
                label="Error Handler Agent",
                data=NodeData(
                    agent_instructions="""Handle errors from risky operations.
                    Use conditional_gate to route based on operation outcome:
                    - success: continue normal processing
                    - error: execute fallback strategy""",
                    tools=["conditional_gate"],
                    config={},
                    ins=["operation_result", "operation_error"],
                    outs=["success_path", "error_path"]
                )
            ),
            NodeSpec(
                id="success_processor",
                type="agent",
                label="Success Processor",
                data=NodeData(
                    agent_instructions="Process successful operation result.",
                    config={},
                    ins=["success_path"],
                    outs=["success_result"]
                )
            ),
            NodeSpec(
                id="fallback_processor",
                type="agent",
                label="Fallback Processor",
                data=NodeData(
                    agent_instructions="Execute fallback strategy for failed operation.",
                    config={},
                    ins=["error_path"],
                    outs=["fallback_result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(id="operation_to_handler", source="risky_operation", target="error_handler", data=EdgeData()),
            EdgeSpec(id="handler_to_success", source="error_handler", target="success_processor", 
                    data=EdgeData(condition="routed_to == 'success_path'")),
            EdgeSpec(id="handler_to_fallback", source="error_handler", target="fallback_processor", 
                    data=EdgeData(condition="routed_to == 'error_path'"))
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Error Handling Test Workflow",
            description="Workflow with error handling and fallback mechanisms",
            nodes=nodes,
            edges=edges,
            metadata={"test_type": "error_handling", "factory": "created"}
        )


class MockDataFactory:
    """Factory for creating mock data for workflow testing."""
    
    @staticmethod
    def create_market_sentiment_data(
        sentiment: str = "bullish",
        confidence: float = 0.8,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create mock market sentiment data."""
        data = {
            "sentiment": sentiment,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "source": "mock_factory"
        }
        if additional_fields:
            data.update(additional_fields)
        return data
    
    @staticmethod
    def create_crypto_price_data(
        symbol: str = "BTC",
        price: float = 50000.0,
        change_24h: float = 2.5,
        volume: str = "high"
    ) -> Dict[str, Any]:
        """Create mock cryptocurrency price data."""
        return {
            "symbol": symbol,
            "price": price,
            "change_24h": change_24h,
            "change_percent_24h": change_24h / price * 100,
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
            "source": "mock_factory"
        }
    
    @staticmethod
    def create_comprehensive_market_data(
        sentiment: str = "bullish",
        confidence: float = 0.8,
        price_change: float = 2.5,
        volume: str = "high",
        volatility: float = 0.3,
        technical_score: float = 0.7
    ) -> Dict[str, Any]:
        """Create comprehensive market data for complex testing."""
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "price_change": price_change,
            "volume": volume,
            "volatility": volatility,
            "technical_score": technical_score,
            "news_sentiment": "positive" if sentiment == "bullish" else "negative",
            "market_cap": 1000.0,
            "liquidity": volume,
            "timestamp": datetime.now().isoformat(),
            "source": "mock_factory"
        }


class TestScenarioBuilder:
    """Builder for creating comprehensive test scenarios."""
    
    def __init__(self):
        self.scenarios = []
    
    def add_conditional_scenario(
        self,
        name: str,
        input_data: Dict[str, Any],
        expected_route: str,
        description: str = ""
    ):
        """Add a conditional routing test scenario."""
        self.scenarios.append({
            "name": name,
            "type": "conditional_routing",
            "input_data": input_data,
            "expected_route": expected_route,
            "description": description
        })
        return self
    
    def add_user_input_scenario(
        self,
        name: str,
        user_input: str,
        expected_processing: str,
        validation_should_pass: bool = True
    ):
        """Add a user input test scenario."""
        self.scenarios.append({
            "name": name,
            "type": "user_input",
            "user_input": user_input,
            "expected_processing": expected_processing,
            "validation_should_pass": validation_should_pass
        })
        return self
    
    def add_error_scenario(
        self,
        name: str,
        error_type: str,
        expected_fallback: str,
        recovery_strategy: str = "default"
    ):
        """Add an error handling test scenario."""
        self.scenarios.append({
            "name": name,
            "type": "error_handling",
            "error_type": error_type,
            "expected_fallback": expected_fallback,
            "recovery_strategy": recovery_strategy
        })
        return self
    
    def build(self) -> List[Dict[str, Any]]:
        """Build and return all scenarios."""
        return self.scenarios.copy()


# Pre-built scenario collections
CRYPTO_TRADING_SCENARIOS = TestScenarioBuilder() \
    .add_conditional_scenario(
        "aggressive_buy",
        MockDataFactory.create_comprehensive_market_data(
            sentiment="bullish", confidence=0.9, technical_score=0.8, volume="high"
        ),
        "aggressive_buy",
        "Strong bullish signals warrant aggressive buying"
    ) \
    .add_conditional_scenario(
        "conservative_sell",
        MockDataFactory.create_comprehensive_market_data(
            sentiment="bearish", confidence=0.7, volatility=0.8
        ),
        "conservative_sell",
        "Moderate bearish signals with high volatility"
    ) \
    .add_conditional_scenario(
        "hold_neutral",
        MockDataFactory.create_comprehensive_market_data(
            sentiment="neutral", confidence=0.5, volatility=0.4
        ),
        "hold",
        "Neutral market conditions suggest holding"
    ) \
    .build()

USER_INPUT_SCENARIOS = TestScenarioBuilder() \
    .add_user_input_scenario(
        "valid_sentiment",
        "sentiment:bullish,confidence:0.8",
        "positive_routing",
        True
    ) \
    .add_user_input_scenario(
        "invalid_format",
        "just some random text",
        "error_handling",
        False
    ) \
    .add_user_input_scenario(
        "edge_case_confidence",
        "sentiment:bullish,confidence:0.7",
        "boundary_processing",
        True
    ) \
    .build()


if __name__ == "__main__":
    # Example usage
    factory = WorkflowFactory()
    
    # Create different types of test workflows
    basic_workflow = factory.create_basic_conditional_workflow()
    crypto_workflow = factory.create_crypto_trading_workflow("ETH")
    user_input_workflow = factory.create_user_input_workflow()
    
    print(f"Created basic workflow with {len(basic_workflow.nodes)} nodes")
    print(f"Created crypto workflow with {len(crypto_workflow.nodes)} nodes")
    print(f"Created user input workflow with {len(user_input_workflow.nodes)} nodes")