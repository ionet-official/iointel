#!/usr/bin/env python3
"""Test the new WorkflowSpec refactor."""

import asyncio
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpecLLM, 
    WorkflowSpec,
    DataSourceNodeLLM,
    AgentNodeLLM,
    DecisionNodeLLM,
    EdgeSpecLLM,
    DataSourceData,
    DataSourceConfig,
    AgentConfig,
    DecisionConfig,
    SLARequirements
)


def test_data_source_node():
    """Test creating a data source node."""
    node = DataSourceNodeLLM(
        type="data_source",
        label="Stock Symbol",
        data=DataSourceData(
            source_name="user_input",
            config=DataSourceConfig(
                message="Enter stock symbol",
                default_value="AAPL"
            )
        )
    )
    print(f"✅ Data source node created: {node.label}")
    return node


def test_agent_node():
    """Test creating an agent node."""
    node = AgentNodeLLM(
        type="agent",
        label="Stock Analyzer",
        data=AgentConfig(
            agent_instructions="Analyze the stock {Stock Symbol} using available tools",
            tools=["get_current_stock_price", "yfinance.get_stock_info"],
            model="gpt-4o",
            sla=SLARequirements(
                tool_usage_required=True,
                required_tools=["get_current_stock_price"],
                min_tool_calls=1,
                enforce_usage=True
            )
        )
    )
    print(f"✅ Agent node created: {node.label}")
    return node


def test_decision_node():
    """Test creating a decision node."""
    node = DecisionNodeLLM(
        type="decision",
        label="Trade Decision",
        data=DecisionConfig(
            agent_instructions="Decide whether to buy or sell based on analysis",
            tools=["conditional_gate"],
            model="gpt-4o",
            sla=SLARequirements(
                tool_usage_required=True,
                required_tools=["conditional_gate"],
                final_tool_must_be="conditional_gate",
                enforce_usage=True
            )
        )
    )
    print(f"✅ Decision node created: {node.label}")
    return node


def test_workflow_spec():
    """Test creating a complete workflow spec."""
    # Create nodes
    data_source = test_data_source_node()
    agent = test_agent_node()
    decision = test_decision_node()
    
    # Create edges  
    edges = [
        EdgeSpecLLM(
            source="Stock Symbol",
            target="Stock Analyzer"
        ),
        EdgeSpecLLM(
            source="Stock Analyzer",
            target="Trade Decision"
        ),
        EdgeSpecLLM(
            source="Trade Decision",
            target="Buy Order",
            route_index=0,
            route_label="buy"
        ),
        EdgeSpecLLM(
            source="Trade Decision",
            target="Sell Order",
            route_index=1,
            route_label="sell"
        )
    ]
    
    # Create workflow spec
    workflow_llm = WorkflowSpecLLM(
        reasoning="Testing the new workflow spec structure",
        title="Stock Trading Workflow",
        description="Analyze stocks and make trading decisions",
        nodes=[data_source, agent, decision],
        edges=edges
    )
    
    print(f"✅ WorkflowSpecLLM created: {workflow_llm.title}")
    
    # Convert to executable spec
    workflow = WorkflowSpec.from_llm_spec(workflow_llm)
    print(f"✅ WorkflowSpec created with ID: {workflow.id}")
    
    # Test validation
    validation_catalog = {
        "user_input": {"description": "User input"},
        "get_current_stock_price": {"description": "Get stock price"},
        "yfinance.get_stock_info": {"description": "Get stock info"},
        "conditional_gate": {"description": "Routing gate"}
    }
    
    issues = workflow.validate_structure(validation_catalog)
    if issues:
        print(f"❌ Validation issues: {issues}")
    else:
        print("✅ Workflow validation passed!")
    
    # Test LLM prompt generation
    prompt = workflow.to_llm_prompt()
    print("\n📝 LLM Prompt representation:")
    print(prompt)
    
    return workflow


async def test_workflow_planner():
    """Test the WorkflowPlanner with new spec."""
    from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
    
    planner = WorkflowPlanner()
    
    # Simple test query
    query = "Create a simple stock analysis workflow"
    
    # Mock tool catalog
    tool_catalog = {
        "get_current_stock_price": {
            "description": "Get current stock price",
            "parameters": {"symbol": "string"}
        },
        "yfinance.get_stock_info": {
            "description": "Get detailed stock information",
            "parameters": {"symbol": "string"}
        }
    }
    
    try:
        print("\n🚀 Testing WorkflowPlanner...")
        workflow = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog,
            max_retries=1
        )
        
        if isinstance(workflow, WorkflowSpec):
            print(f"✅ WorkflowPlanner generated: {workflow.title}")
            print(f"   Nodes: {len(workflow.nodes)}")
            print(f"   Edges: {len(workflow.edges)}")
        else:
            print(f"✅ Chat-only response: {workflow.reasoning}")
            
    except Exception as e:
        print(f"❌ WorkflowPlanner error: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New WorkflowSpec Structure")
    print("=" * 60)
    
    # Test individual components
    workflow = test_workflow_spec()
    
    # Test workflow planner
    print("\n" + "=" * 60)
    asyncio.run(test_workflow_planner())
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()