#!/usr/bin/env python3
"""
Test Data Source vs Agent Tool Confusion
========================================

This test demonstrates and validates the conceptual confusion where the workflow
planner tries to use prompt_tool (a data_source) to fetch external data like
stock prices, when it should be using agent nodes with appropriate tools.

The issue: prompt_tool is for injecting context/prompts, NOT for fetching data!
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from iointel.src.utilities.workflow_test_repository import WorkflowTestRepository, TestLayer, WorkflowTestCase
from iointel.src.utilities.workflow_helpers import generate_only
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog

async def add_data_source_confusion_tests():
    """Add tests that validate proper data_source vs agent node usage."""
    
    # Setup
    load_tools_from_env()
    tool_catalog = create_tool_catalog()
    repo = WorkflowTestRepository(storage_dir="smart_test_repository")
    
    print("🧪 ADDING DATA SOURCE CONFUSION TESTS")
    print("=" * 50)
    
    # Test 1: Logical test - validate that prompt_tool data sources don't fetch external data
    test1 = repo.create_logical_test(
        name="Data Source Purpose Validation",
        description="Validates that data_source nodes are only used for input collection, not data fetching",
        category="data_source_validation",
        workflow_spec={
            "title": "Invalid Data Fetching with prompt_tool",
            "description": "Incorrectly tries to fetch stock data with prompt_tool data source",
            "version": "1.0",
            "nodes": [
                {
                    "id": "stock_data_1",
                    "type": "data_source",
                    "label": "Fetch Stock Data",
                    "data": {
                        "source_name": "prompt_tool",
                        "config": {"message": "Get AAPL stock price"}  # WRONG! prompt_tool doesn't fetch data
                    }
                }
            ],
            "edges": []
        },
        expected_result={
            "validation_passed": False,
            "error_contains": "prompt_tool"
        },
        should_pass=False,  # This SHOULD fail validation
        tags=["data_source", "prompt_tool", "validation", "anti-pattern"]
    )
    print(f"✅ Added test: {test1.name}")
    
    # Test 2: Agentic test - ensure stock workflows use agent nodes
    test2 = repo.create_agentic_test(
        name="Stock Workflow Generation Pattern",
        description="Validates that stock price workflows use agent nodes with tools, not data_source nodes",
        category="workflow_generation",
        user_prompt="Create a workflow to get current Bitcoin price and historical price from 1 year ago",
        expected_result={
            "workflow_generated": True,
            "has_agent_with_coin_tools": True,  # Should use agent with get_coin_quotes tools
            "no_prompt_tool_for_data": True,     # Should NOT use prompt_tool to fetch prices
            "correct_tool_usage": True
        },
        should_pass=True,
        tags=["data_source", "agent_tools", "crypto", "generation"]
    )
    print(f"✅ Added test: {test2.name}")
    
    # Test 3: Validate correct data_source usage
    test3 = repo.create_logical_test(
        name="Correct prompt_tool Usage",
        description="Validates correct usage of prompt_tool for context injection",
        category="data_source_validation",
        workflow_spec={
            "title": "Correct prompt_tool Usage",
            "description": "Uses prompt_tool correctly for context injection",
            "version": "1.0",
            "nodes": [
                {
                    "id": "context_1",
                    "type": "data_source",
                    "label": "Initial Context",
                    "data": {
                        "source_name": "prompt_tool",
                        "config": {"message": "You are analyzing stocks. Be thorough."}
                    }
                },
                {
                    "id": "analyzer_1",
                    "type": "agent",
                    "label": "Stock Analyzer",
                    "data": {
                        "agent_instructions": "Analyze the stock market data",
                        "tools": ["get_coin_quotes", "get_coin_quotes_historical"],
                        "ins": ["context"]
                    }
                }
            ],
            "edges": [
                {"source": "context_1", "target": "analyzer_1"}
            ]
        },
        expected_result={
            "validation_passed": True,
            "proper_separation": True  # Context injection separate from data fetching
        },
        should_pass=True,
        tags=["data_source", "prompt_tool", "best_practice"]
    )
    print(f"✅ Added test: {test3.name}")
    
    # Test 4: Complex stock workflow pattern
    test4 = repo.create_agentic_test(
        name="Stock Decision Workflow Pattern", 
        description="Full stock decision workflow should use agents for data, not data_sources",
        category="workflow_generation",
        user_prompt="Create a stock trading decision workflow with 5% threshold comparing current vs 1 year ago prices, routing to buy/sell agents",
        expected_result={
            "workflow_generated": True,
            "decision_node_has_tools": True,  # Decision node should have price tools
            "no_data_source_for_prices": True,  # No data_source nodes fetching prices
            "has_user_input": True,  # Should start with user_input for stock symbol
            "proper_tool_assignment": True
        },
        should_pass=True,
        tags=["data_source", "agent_tools", "routing", "decision", "stock_trading"]
    )
    print(f"✅ Added test: {test4.name}")
    
    print(f"\n✅ Successfully added 4 data source confusion tests!")
    print("Run with: python run_unified_tests.py --tags data_source")

async def validate_current_behavior():
    """Quick validation of current workflow generation behavior."""
    print("\n📊 VALIDATING CURRENT BEHAVIOR")
    print("=" * 50)
    
    load_tools_from_env()
    tool_catalog = create_tool_catalog()
    
    # Test what happens with stock workflow
    spec = await generate_only(
        "Get current Bitcoin price",
        tool_catalog
    )
    
    if spec:
        print(f"✅ Generated workflow: {spec.title}")
        
        # Check for data_source nodes
        data_sources = [n for n in spec.nodes if n.type == "data_source"]
        print(f"\n📋 Data source nodes: {len(data_sources)}")
        for ds in data_sources:
            source_name = ds.data.source_name if hasattr(ds.data, 'source_name') else 'unknown'
            print(f"  - {ds.label} (source: {source_name})")
        
        # Check for agent nodes with tools
        agents = [n for n in spec.nodes if n.type == "agent"]
        print(f"\n🤖 Agent nodes: {len(agents)}")
        for agent in agents:
            tools = agent.data.tools if hasattr(agent.data, 'tools') else []
            print(f"  - {agent.label} (tools: {tools})")
        
        # Flag issues
        if any(ds for ds in data_sources if hasattr(ds.data, 'source_name') and 
               ds.data.source_name == 'prompt_tool' and 'price' in ds.label.lower()):
            print("\n⚠️ WARNING: Found prompt_tool being used for price fetching!")
    else:
        print("❌ Failed to generate workflow")

if __name__ == "__main__":
    asyncio.run(add_data_source_confusion_tests())
    # asyncio.run(validate_current_behavior())