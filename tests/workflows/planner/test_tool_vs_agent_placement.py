import os
#!/usr/bin/env python3
"""
Test WorkflowPlanner Tool vs Agent Node Placement
===============================================
Tests the critical fix for tool vs agent node placement based on user requirements:
1. When user asks for "agent using tools X, Y, Z", should create ONE agent node with those tools
2. Should NOT create separate tool nodes for external APIs
3. Tool nodes should only be used for user_input, simple math with hardcoded values
4. Tests the standardized workflow generation reporting system
"""
import asyncio
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
import os


class TestToolVsAgentPlacement:
    """Test cases for tool vs agent node placement fixes."""
    
    @staticmethod
    async def test_stock_decision_agent_with_tools():
        """Test the specific case from the user's screenshot - stock decision agent using tools."""
        print("\n🧪 TEST: Stock Decision Agent with Tools")
        print("=" * 50)
        
        # Load tools first to populate the catalog
        load_tools_from_env("creds.env")
        
        planner = WorkflowPlanner()
        tool_catalog = create_tool_catalog()
        
        # This is the exact type of query that was creating separate tool nodes
        query = """A user input, connected to a stock Decision agent using tools that 
        fetch historical and current stock prices, with a required conditional 
        gate that connects to a buy or sell agent. A trade is triggered if the 
        given stock(s) are 5% greater or less than their historical price (user 
        will inform when, and if not, just compare to yesterday). A 5% bump 
        means a sell, a -5% or more means a buy."""
        
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print(f"📋 Generated: {workflow_spec.title}")
        print(f"📊 Nodes: {len(workflow_spec.nodes)}")
        
        # Critical analysis - should NOT have separate tool nodes for stock price APIs
        tool_nodes = [n for n in workflow_spec.nodes if n.type == 'tool']
        stock_tool_nodes = [n for n in tool_nodes 
                           if hasattr(n.data, 'tool_name') and 
                           ('stock' in n.data.tool_name.lower() or 'price' in n.data.tool_name.lower())]
        
        print("\n🔍 ANALYSIS - Tool Nodes:")
        for node in tool_nodes:
            tool_name = getattr(node.data, 'tool_name', 'unknown')
            print(f"  🔧 {node.label} (tool: {tool_name})")
            if tool_name not in ['user_input', 'add', 'subtract', 'multiply', 'divide']:
                print(f"    ⚠️  PROBLEMATIC: '{tool_name}' should likely be in an agent node!")
        
        print("\n🔍 ANALYSIS - Agent Nodes:")
        decision_agents = [n for n in workflow_spec.nodes if n.type == 'decision']
        for agent in decision_agents:
            tools = getattr(agent.data, 'tools', [])
            print(f"  🎯 {agent.label} with tools: {tools}")
            
            # Check if decision agent has stock price tools
            stock_tools_in_agent = [t for t in tools if 'stock' in t.lower() or 'price' in t.lower()]
            if stock_tools_in_agent:
                print(f"    ✅ GOOD: Decision agent includes stock tools: {stock_tools_in_agent}")
        
        # The fix should result in:
        # 1. At most one user_input tool node
        # 2. NO separate tool nodes for stock price APIs
        # 3. Decision agent(s) WITH stock price tools
        
        assert len(stock_tool_nodes) == 0, f"❌ FAILED: Found {len(stock_tool_nodes)} stock API tool nodes, should be 0!"
        assert len(decision_agents) > 0, "❌ FAILED: No decision agents found!"
        
        has_stock_tools_in_agents = any(
            any('stock' in t.lower() or 'price' in t.lower() 
                for t in getattr(agent.data, 'tools', []))
            for agent in decision_agents
        )
        assert has_stock_tools_in_agents, "❌ FAILED: No decision agents have stock price tools!"
        
        print("✅ PASSED: Stock price tools properly placed in agent nodes, not as separate tool nodes!")
        return workflow_spec
    
    @staticmethod
    async def test_data_fetcher_with_external_apis():
        """Test that data fetcher agents are created instead of tool nodes for external APIs."""
        print("\n🧪 TEST: Data Fetcher with External APIs")
        print("=" * 50)
        
        # Load tools first to populate the catalog
        load_tools_from_env("creds.env")
        
        planner = WorkflowPlanner()
        tool_catalog = create_tool_catalog()
        
        query = "Get current weather for New York and compare it with historical data, then analyze the trend"
        
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print(f"📋 Generated: {workflow_spec.title}")
        
        # Should have data_fetcher or analyzer agents, not tool nodes for weather APIs
        tool_nodes = [n for n in workflow_spec.nodes if n.type == 'tool']
        weather_tool_nodes = [n for n in tool_nodes 
                             if hasattr(n.data, 'tool_name') and 'weather' in n.data.tool_name.lower()]
        
        data_fetcher_nodes = [n for n in workflow_spec.nodes if n.type == 'data_fetcher']
        analyzer_nodes = [n for n in workflow_spec.nodes if n.type == 'analyzer']
        
        print("\n📊 Results:")
        print(f"  Tool nodes with weather APIs: {len(weather_tool_nodes)}")
        print(f"  Data fetcher nodes: {len(data_fetcher_nodes)}")
        print(f"  Analyzer nodes: {len(analyzer_nodes)}")
        
        assert len(weather_tool_nodes) == 0, "❌ FAILED: Weather APIs should be in agent nodes, not tool nodes!"
        assert (len(data_fetcher_nodes) + len(analyzer_nodes)) > 0, "❌ FAILED: Should have data fetcher or analyzer agents!"
        
        print("✅ PASSED: External APIs properly placed in agent nodes!")
        return workflow_spec
    
    @staticmethod
    async def test_valid_tool_node_usage():
        """Test cases where tool nodes ARE appropriate."""
        print("\n🧪 TEST: Valid Tool Node Usage")
        print("=" * 50)
        
        # Load tools first to populate the catalog
        load_tools_from_env("creds.env")
        
        planner = WorkflowPlanner()
        tool_catalog = create_tool_catalog()
        
        query = "Get user input for two numbers, then add them together"
        
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print(f"📋 Generated: {workflow_spec.title}")
        
        # Should have user_input tool nodes (valid) and possibly math tool nodes
        tool_nodes = [n for n in workflow_spec.nodes if n.type == 'tool']
        user_input_nodes = [n for n in tool_nodes 
                           if hasattr(n.data, 'tool_name') and n.data.tool_name == 'user_input']
        
        print(f"\n📊 Tool nodes found: {len(tool_nodes)}")
        for node in tool_nodes:
            tool_name = getattr(node.data, 'tool_name', 'unknown')
            is_valid = tool_name in ['user_input', 'add', 'subtract', 'multiply', 'divide']
            status = "✅ VALID" if is_valid else "⚠️  QUESTIONABLE"
            print(f"  {status}: {node.label} (tool: {tool_name})")
        
        assert len(user_input_nodes) > 0, "❌ FAILED: Should have user_input tool nodes!"
        print("✅ PASSED: Appropriate use of tool nodes for user input!")
        return workflow_spec
    
    @staticmethod
    async def test_generation_reporting():
        """Test the standardized workflow generation reporting system."""
        print("\n🧪 TEST: Generation Reporting System")
        print("=" * 50)
        
        # Load tools first to populate the catalog
        load_tools_from_env("creds.env")
        
        planner = WorkflowPlanner()
        tool_catalog = create_tool_catalog()
        
        # This query should trigger problematic tool node creation for visibility testing
        query = "Use the get_current_stock_price tool to fetch Apple stock data"
        
        print("🔍 Testing reporting system (check logs for standardized reports)...")
        
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        # The reporting system should automatically log analysis using to_llm_prompt()
        # This is logged internally via system_logger.info()
        
        # Test that we can generate the same report manually
        temp_report = planner._create_workflow_generation_report(
            query=query,
            attempt=1,
            max_retries=3,
            workflow_spec=workflow_spec
        )
        
        assert "WORKFLOW GENERATION ANALYSIS" in temp_report
        assert query in temp_report
        assert workflow_spec.title in temp_report
        assert "Single Source of Truth" in temp_report  # Uses to_llm_prompt()
        
        # Check if it flags problematic tool nodes
        tool_nodes = [n for n in workflow_spec.nodes if n.type == 'tool']
        stock_tool_nodes = [n for n in tool_nodes 
                           if hasattr(n.data, 'tool_name') and 'stock' in n.data.tool_name.lower()]
        
        if stock_tool_nodes:
            assert "TOOL NODES DETECTED" in temp_report
            assert "REVIEW" in temp_report
        
        print("✅ PASSED: Generation reporting system working correctly!")
        print(f"📄 Report length: {len(temp_report)} characters")
        return temp_report


async def run_all_tests():
    """Run all tool vs agent placement tests."""
    print("🚀 RUNNING TOOL vs AGENT NODE PLACEMENT TESTS")
    print("=" * 60)
    
    test_suite = TestToolVsAgentPlacement()
    results = {}
    
    try:
        # Test 1: Stock decision agent (the main case from user's screenshot)
        results['stock_decision'] = await test_suite.test_stock_decision_agent_with_tools()
        
        # Test 2: Data fetcher with external APIs
        results['data_fetcher'] = await test_suite.test_data_fetcher_with_external_apis()
        
        # Test 3: Valid tool node usage
        results['valid_tool_usage'] = await test_suite.test_valid_tool_node_usage()
        
        # Test 4: Generation reporting system
        results['reporting'] = await test_suite.test_generation_reporting()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("✅ Tool vs Agent node placement is working correctly")
        print("✅ Standardized reporting system is functioning") 
        print("✅ WorkflowPlanner prompt fixes are effective")
        
        return results
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting Tool vs Agent Node Placement Tests...")
    results = asyncio.run(run_all_tests())
    
    if results:
        print("\n📊 SUMMARY:")
        print(f"  Stock Decision Test: {'✅' if results['stock_decision'] else '❌'}")
        print(f"  Data Fetcher Test: {'✅' if results['data_fetcher'] else '❌'}")
        print(f"  Valid Tool Usage Test: {'✅' if results['valid_tool_usage'] else '❌'}")
        print(f"  Reporting System Test: {'✅' if results['reporting'] else '❌'}")
    else:
        print("❌ Tests failed - see output above")
        sys.exit(1)