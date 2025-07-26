#!/usr/bin/env python3
"""
Smart Test Repository - EXECUTION ENGINE
========================================

This demonstrates how the smart test repository actually RUNS tests
and validates against expected_result configurations.

The storage was just part 1. This is part 2: the execution engine.
"""

import asyncio
from typing import Dict, Any, List
from uuid import uuid4

from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository, 
    TestLayer, 
    WorkflowTestCase
)
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.agent_methods.data_models.data_source_registry import get_valid_data_source_names


class SmartTestRunner:
    """Execution engine for the smart test repository."""
    
    def __init__(self, repository: WorkflowTestRepository):
        self.repo = repository
        self.results = {}
    
    def run_logical_test(self, test_case: WorkflowTestCase) -> Dict[str, Any]:
        """
        Run a logical layer test and validate against expected_result.
        
        This tests pure data structures - no LLM calls.
        """
        print(f"🧪 RUNNING LOGICAL TEST: {test_case.name}")
        
        try:
            # 1. Load the workflow spec
            workflow = WorkflowSpec(**test_case.workflow_spec)
            print(f"   ✅ Loaded workflow: {workflow.title}")
            
            # 2. Create tool catalog for validation
            mock_catalog = {
                "user_input": {"description": "Get user input"},
                "Crawler-scrape_url": {"description": "Scrape URL content"},
                "get_current_stock_price": {"description": "Get stock price"},
                "conditional_gate": {"description": "Route based on conditions"},
                "send_email": {"description": "Send email"}
            }
            
            # 3. Run validation
            issues = workflow.validate_structure(mock_catalog)
            
            # 4. Analyze results against expected_result
            actual_result = self._analyze_logical_workflow(workflow, issues, mock_catalog)
            expected = test_case.expected_result or {}
            
            # 5. Compare actual vs expected
            validation_results = self._validate_results(actual_result, expected)
            
            print(f"   📊 VALIDATION RESULTS:")
            for key, result in validation_results.items():
                status = "✅" if result['passed'] else "❌"
                print(f"      {status} {key}: expected={result['expected']}, actual={result['actual']}")
            
            # 6. Determine overall pass/fail
            all_passed = all(r['passed'] for r in validation_results.values())
            overall_passed = all_passed and (len(issues) == 0) == test_case.should_pass
            
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': overall_passed,
                'validation_issues': issues,
                'actual_result': actual_result,
                'expected_result': expected,
                'detailed_results': validation_results
            }
            
        except Exception as e:
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': False,
                'error': str(e)
            }
    
    async def run_agentic_test(self, test_case: WorkflowTestCase) -> Dict[str, Any]:
        """
        Run an agentic layer test and validate against expected_result.
        
        This tests LLM workflow generation.
        """
        print(f"🤖 RUNNING AGENTIC TEST: {test_case.name}")
        
        try:
            # 1. Initialize WorkflowPlanner
            planner = WorkflowPlanner()
            
            # 2. Generate workflow from user prompt
            print(f"   💭 User prompt: '{test_case.user_prompt}'")
            workflow = await planner.generate_workflow(
                query=test_case.user_prompt,
                tool_catalog=test_case.tool_catalog or {}
            )
            
            # 3. Analyze the generated workflow
            actual_result = self._analyze_agentic_workflow(workflow, test_case.tool_catalog or {})
            expected = test_case.expected_result or {}
            
            # 4. Compare actual vs expected
            validation_results = self._validate_results(actual_result, expected)
            
            print(f"   📊 GENERATION RESULTS:")
            for key, result in validation_results.items():
                status = "✅" if result['passed'] else "❌"
                print(f"      {status} {key}: expected={result['expected']}, actual={result['actual']}")
            
            # 5. Determine overall pass/fail
            all_passed = all(r['passed'] for r in validation_results.values())
            
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': all_passed,
                'generated_workflow': {
                    'title': workflow.title,
                    'nodes': len(workflow.nodes),
                    'edges': len(workflow.edges)
                },
                'actual_result': actual_result,
                'expected_result': expected,
                'detailed_results': validation_results
            }
            
        except Exception as e:
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': False,
                'error': str(e)
            }
    
    async def run_orchestration_test(self, test_case: WorkflowTestCase) -> Dict[str, Any]:
        """
        Run an orchestration layer test and validate against expected_result.
        
        This tests full pipeline execution.
        """
        print(f"🎯 RUNNING ORCHESTRATION TEST: {test_case.name}")
        
        try:
            # 1. Generate workflow from prompt (if needed)
            if test_case.user_prompt:
                planner = WorkflowPlanner()
                workflow = await planner.generate_workflow(
                    query=test_case.user_prompt,
                    tool_catalog=test_case.tool_catalog or {}
                )
            else:
                workflow = WorkflowSpec(**test_case.workflow_spec)
            
            # 2. Simulate DAG execution (mock for demo)
            execution_result = self._simulate_dag_execution(workflow)
            
            # 3. Analyze execution results
            actual_result = self._analyze_orchestration_results(workflow, execution_result)
            expected = test_case.expected_result or {}
            
            # 4. Compare actual vs expected
            validation_results = self._validate_results(actual_result, expected)
            
            print(f"   📊 EXECUTION RESULTS:")
            for key, result in validation_results.items():
                status = "✅" if result['passed'] else "❌"
                print(f"      {status} {key}: expected={result['expected']}, actual={result['actual']}")
            
            # 5. Determine overall pass/fail
            all_passed = all(r['passed'] for r in validation_results.values())
            
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': all_passed,
                'execution_result': execution_result,
                'actual_result': actual_result,
                'expected_result': expected,
                'detailed_results': validation_results
            }
            
        except Exception as e:
            return {
                'test_case_id': test_case.id,
                'test_name': test_case.name,
                'passed': False,
                'error': str(e)
            }
    
    def _analyze_logical_workflow(self, workflow: WorkflowSpec, issues: List[str], tool_catalog: Dict) -> Dict[str, Any]:
        """Analyze a workflow for logical test validation."""
        data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
        agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
        
        return {
            'validates_successfully': len(issues) == 0,
            'no_invalid_source_names': all(
                getattr(node.data, 'source_name', None) in get_valid_data_source_names()
                for node in data_source_nodes
            ),
            'web_crawling_uses_agent_tools': any(
                hasattr(node.data, 'tools') and node.data.tools and
                any('crawler' in tool.lower() or 'scrape' in tool.lower() for tool in node.data.tools)
                for node in agent_nodes
                if 'crawl' in node.label.lower() or 'web' in node.label.lower()
            ),
            'has_data_source_nodes': len(data_source_nodes) > 0,
            'has_agent_nodes': len(agent_nodes) > 0
        }
    
    def _analyze_agentic_workflow(self, workflow: WorkflowSpec, tool_catalog: Dict) -> Dict[str, Any]:
        """Analyze a generated workflow for agentic test validation."""
        data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
        agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
        decision_nodes = [n for n in workflow.nodes if n.type == 'decision']
        
        # Check for stock-related tools in agents
        agent_nodes_have_stock_tools = any(
            hasattr(node.data, 'tools') and node.data.tools and
            any('stock' in tool.lower() or 'price' in tool.lower() or 'yfinance' in tool.lower()
                for tool in node.data.tools)
            for node in agent_nodes
        )
        
        # Check for decision agent with stock tools
        has_decision_agent_with_stock_tools = any(
            hasattr(node.data, 'tools') and node.data.tools and
            any('stock' in tool.lower() or 'price' in tool.lower() for tool in node.data.tools)
            for node in decision_nodes
        )
        
        # Check for conditional routing
        has_conditional_routing = len(decision_nodes) > 0 and any(
            edge.data.condition or edge.data.route_index is not None
            for edge in workflow.edges
        )
        
        # Check for email capabilities
        buy_sell_agents_have_email = any(
            hasattr(node.data, 'tools') and node.data.tools and
            any('email' in tool.lower() for tool in node.data.tools)
            for node in agent_nodes
            if 'buy' in node.label.lower() or 'sell' in node.label.lower()
        )
        
        return {
            'has_data_source_nodes': len(data_source_nodes) > 0,
            'has_agent_nodes': len(agent_nodes) > 0,
            'agent_nodes_have_stock_tools': agent_nodes_have_stock_tools,
            'creates_agent_with_crawler_tool': any(
                hasattr(node.data, 'tools') and node.data.tools and
                any('crawler' in tool.lower() or 'scrape' in tool.lower() for tool in node.data.tools)
                for node in agent_nodes
            ),
            'does_not_create_crawl_web_data_source': not any(
                getattr(node.data, 'source_name', '') == 'crawl_the_web'
                for node in data_source_nodes
            ),
            'uses_valid_source_names_only': all(
                getattr(node.data, 'source_name', None) in get_valid_data_source_names()
                for node in data_source_nodes
            ),
            'has_decision_agent_with_stock_tools': has_decision_agent_with_stock_tools,
            'has_conditional_routing': has_conditional_routing,
            'buy_sell_agents_have_email': buy_sell_agents_have_email
        }
    
    def _analyze_orchestration_results(self, workflow: WorkflowSpec, execution_result: Dict) -> Dict[str, Any]:
        """Analyze orchestration execution results."""
        return {
            'workflow_executes_successfully': execution_result.get('status') == 'success',
            'sla_enforcement_applied': execution_result.get('sla_enforced', False),
            'conditional_routing_works': execution_result.get('routing_executed', False),
            'emails_sent_successfully': execution_result.get('emails_sent', 0) > 0,
            'stock_prices_fetched': execution_result.get('stock_data_retrieved', False),
            'has_user_input_data_source': any(
                n.type == 'data_source' and getattr(n.data, 'source_name', None) == 'user_input'
                for n in workflow.nodes
            )
        }
    
    def _simulate_dag_execution(self, workflow: WorkflowSpec) -> Dict[str, Any]:
        """Simulate DAG execution for demo purposes."""
        # In real implementation, this would use DAGExecutor
        return {
            'status': 'success',
            'nodes_executed': len(workflow.nodes),
            'sla_enforced': any(hasattr(n, 'sla') and n.sla for n in workflow.nodes),
            'routing_executed': any(n.type == 'decision' for n in workflow.nodes),
            'emails_sent': 1,  # Simulated
            'stock_data_retrieved': any(
                hasattr(n.data, 'tools') and n.data.tools and
                any('stock' in tool.lower() for tool in n.data.tools)
                for n in workflow.nodes
            )
        }
    
    def _validate_results(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Dict]:
        """Compare actual results against expected results."""
        validation_results = {}
        
        for key, expected_value in expected.items():
            actual_value = actual.get(key, None)
            passed = actual_value == expected_value
            
            validation_results[key] = {
                'expected': expected_value,
                'actual': actual_value,
                'passed': passed
            }
        
        return validation_results


async def demonstrate_test_execution():
    """Demonstrate the full execution engine."""
    print("🚀 SMART TEST REPOSITORY - EXECUTION ENGINE")
    print("=" * 60)
    
    # 1. Initialize repository and runner
    repo = WorkflowTestRepository(storage_dir="smart_test_repository")
    runner = SmartTestRunner(repo)
    
    # 2. Add a comprehensive test case
    test_case = repo.create_logical_test(
        name="Complete Data Source Validation",
        description="Tests that workflow correctly uses data sources and agent tools",
        category="data_source_validation",
        workflow_spec={
            "id": str(uuid4()),
            "rev": 1,
            "title": "Data Source Validation Test",
            "description": "Test workflow with proper data source usage", 
            "nodes": [
                {
                    "id": "user_input_1",
                    "type": "data_source",
                    "label": "User Input",
                    "data": {
                        "source_name": "user_input",  # Valid!
                        "config": {"prompt": "Enter your query"},
                        "ins": [],
                        "outs": ["query"]
                    }
                },
                {
                    "id": "agent_1",
                    "type": "agent",
                    "label": "Web Crawler Agent",
                    "data": {
                        "agent_instructions": "Crawl the web for information",
                        "tools": ["Crawler-scrape_url"],  # Correct: agent with tools!
                        "ins": ["query"],
                        "outs": ["result"]
                    }
                }
            ],
            "edges": [
                {
                    "id": "edge_1",
                    "source": "user_input_1",
                    "target": "agent_1",
                    "data": {"route_index": 0}
                }
            ]
        },
        expected_result={
            'validates_successfully': True,          # Should validate without issues
            'no_invalid_source_names': True,        # All source names should be valid
            'web_crawling_uses_agent_tools': True,  # Web crawling should use agent tools
            'has_data_source_nodes': True,          # Should have data source nodes
            'has_agent_nodes': True                 # Should have agent nodes
        },
        should_pass=True,
        tags=["execution_demo", "validation", "data_source"]
    )
    
    # 3. Run the logical test
    print("\n1. 🧪 RUNNING LOGICAL TEST")
    logical_result = runner.run_logical_test(test_case)
    print(f"   Overall Result: {'✅ PASSED' if logical_result['passed'] else '❌ FAILED'}")
    
    # 4. Create and run agentic test
    print("\n2. 🤖 RUNNING AGENTIC TEST")
    agentic_test = repo.create_agentic_test(
        name="Anti-Hallucination Generation Test",
        description="Tests that generated workflows don't hallucinate invalid data sources",
        category="anti_hallucination",
        user_prompt="crawl the web tool",  # This should NOT create a data_source
        tool_catalog={
            "user_input": {"description": "Get user input"},
            "Crawler-scrape_url": {"description": "Scrape URL content"}
        },
        expected_result={
            'creates_agent_with_crawler_tool': True,           # Should create agent with crawler
            'does_not_create_crawl_web_data_source': True,     # Should NOT create invalid data source
            'uses_valid_source_names_only': True,              # Only valid source names
            'has_agent_nodes': True                            # Should have agent nodes
        }
    )
    
    agentic_result = await runner.run_agentic_test(agentic_test)
    print(f"   Overall Result: {'✅ PASSED' if agentic_result['passed'] else '❌ FAILED'}")
    
    # 5. Create and run orchestration test
    print("\n3. 🎯 RUNNING ORCHESTRATION TEST")
    orchestration_test = WorkflowTestCase(
        id=str(uuid4()),
        name="Stock Trading Pipeline Execution",
        description="Full execution test of stock trading workflow",
        layer=TestLayer.ORCHESTRATION,
        category="stock_trading_pipeline",
        user_prompt="A user input, connected to a stock Decision agent using tools that fetch historical and current stock prices, with a required conditional gate that connects to a buy or sell agent. A trade is triggered if the given stock(s) are 5% greater or less than their historical price (compare to yesterday). A 5% bump means a sell, a -5% or more means a buy. Both agents are connected to an email agent that sends email to me, alex@io.net about the trade.",
        tool_catalog={
            "user_input": {"description": "Get user input"},
            "get_current_stock_price": {"description": "Get current stock price"},
            "conditional_gate": {"description": "Route based on conditions"},
            "send_email": {"description": "Send email notifications"}
        },
        expected_result={
            'workflow_executes_successfully': True,    # Should execute without errors
            'sla_enforcement_applied': True,           # SLA should be enforced
            'conditional_routing_works': True,         # Routing should work
            'emails_sent_successfully': True,          # Emails should be sent
            'stock_prices_fetched': True,              # Stock data should be retrieved
            'has_user_input_data_source': True        # Should have user input
        }
    )
    
    repo.add_test_case(orchestration_test)
    orchestration_result = await runner.run_orchestration_test(orchestration_test)
    print(f"   Overall Result: {'✅ PASSED' if orchestration_result['passed'] else '❌ FAILED'}")
    
    # 6. Summary
    print("\n📊 EXECUTION SUMMARY")
    print("=" * 30)
    total_tests = 3
    passed_tests = sum([
        logical_result['passed'],
        agentic_result['passed'], 
        orchestration_result['passed']
    ])
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # 7. Show how results are stored
    print(f"\n💾 RESULTS STORAGE")
    runner.results = {
        'logical': logical_result,
        'agentic': agentic_result,
        'orchestration': orchestration_result
    }
    
    print("   ✅ Test results can be stored, analyzed, and reported")
    print("   ✅ Expected vs actual comparisons are automatically validated")
    print("   ✅ Detailed failure analysis is provided")
    
    return runner.results


if __name__ == "__main__":
    asyncio.run(demonstrate_test_execution())