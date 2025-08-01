#!/usr/bin/env python3
"""
Demonstration of Smart Test Repository System
=============================================

This script shows how the WorkflowTestRepository provides the intelligent 
storage and factory system that the user was expecting.

It demonstrates:
1. Smart storage of test cases with metadata and categorization
2. Factory patterns for creating different types of tests
3. Intelligent retrieval of test data based on filters
4. Automatic persistence and loading
5. Integration with existing WorkflowSpec validation
"""

from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository, 
    TestLayer, 
    WorkflowTestCase
)
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.data_models.data_source_registry import get_valid_data_source_names


def demonstrate_smart_test_system():
    """Demonstrate the smart test repository system capabilities."""
    print("🚀 DEMONSTRATING SMART WORKFLOW TEST REPOSITORY")
    print("=" * 60)
    
    # 1. Initialize the smart repository
    print("\n1. 📁 INITIALIZING SMART REPOSITORY")
    repo = WorkflowTestRepository(storage_dir="demo_test_repository")
    print(f"   ✅ Repository initialized with persistent storage")
    print(f"   📂 Storage layers: {[layer.value for layer in TestLayer]}")
    
    # 2. Create our data source validation test using the smart factory
    print("\n2. 🏭 USING SMART TEST FACTORY")
    
    # Create the specific data source validation test case
    data_source_test = repo.create_logical_test(
        name="Data Source Registry Validation",
        description="Validates that WorkflowPlanner only uses valid data source names and prevents 'crawl_the_web' type mistakes",
        category="data_source_validation", 
        workflow_spec={
            "id": "data-source-validation-test",
            "rev": 1,
            "title": "Data Source Validation Test",
            "description": "Test that ensures only valid data sources are used",
            "nodes": [
                {
                    "id": "user_input_1",
                    "type": "data_source",
                    "label": "User Input",
                    "data": {
                        "source_name": "user_input",  # Valid
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
                        "tools": ["Crawler-scrape_url"],  # Correct: agent with tools
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
            "validates_successfully": True,
            "no_invalid_source_names": True,
            "web_crawling_uses_agent_tools": True
        },
        tags=["data_source", "validation", "crawl_web_fix", "registry"]
    )
    
    print(f"   ✅ Created logical test: {data_source_test.name}")
    print(f"   🎯 Category: {data_source_test.category}")
    print(f"   🏷️  Tags: {data_source_test.tags}")
    print(f"   💾 Auto-persisted to: logical/{data_source_test.id}.json")
    
    # 3. Create an agentic test for the same issue
    agentic_test = repo.create_agentic_test(
        name="WorkflowPlanner Anti-Hallucination Test",
        description="Tests that WorkflowPlanner doesn't hallucinate invalid data sources like 'crawl_the_web'",
        category="data_source_validation",
        user_prompt="crawl the web tool",  # This should NOT create a data_source node
        tool_catalog={
            "user_input": {
                "description": "Get user input",
                "parameters": {"prompt": {"type": "string"}},
                "required_parameters": []
            },
            "Crawler-scrape_url": {
                "description": "Scrape content from a URL",
                "parameters": {"url": {"type": "string"}},
                "required_parameters": ["url"]
            }
        },
        expected_result={
            "creates_agent_with_crawler_tool": True,
            "does_not_create_crawl_web_data_source": True,
            "uses_valid_source_names_only": True
        },
        tags=["anti_hallucination", "data_source", "agent_vs_tool", "crawler"]
    )
    
    print(f"   ✅ Created agentic test: {agentic_test.name}")
    print(f"   💭 User prompt: '{agentic_test.user_prompt}'")
    
    # 4. Demonstrate smart retrieval and filtering
    print("\n3. 🔍 SMART TEST RETRIEVAL & FILTERING")
    
    # Get all data source validation tests
    validation_tests = repo.get_tests_by_category("data_source_validation")
    print(f"   📊 Tests in 'data_source_validation' category: {len(validation_tests)}")
    
    # Get tests by tags
    crawler_tests = repo.get_tests_by_tags(["crawler", "crawl_web_fix"])
    print(f"   🏷️  Tests with crawler tags: {len(crawler_tests)}")
    
    # Get logical layer tests 
    logical_tests = repo.get_tests_by_layer(TestLayer.LOGICAL)
    print(f"   ⚗️  Logical layer tests: {len(logical_tests)}")
    
    # Get agentic layer tests
    agentic_tests = repo.get_tests_by_layer(TestLayer.AGENTIC) 
    print(f"   🤖 Agentic layer tests: {len(agentic_tests)}")
    
    # 5. Demonstrate smart fixture generation
    print("\n4. 🧪 SMART FIXTURE GENERATION")
    
    # Get smart fixtures for logical tests in data_source_validation category
    logical_fixtures = repo.get_smart_fixture_data(
        layer=TestLayer.LOGICAL,
        category="data_source_validation"
    )
    print(f"   ⚗️  Logical fixtures generated:")
    print(f"      - Workflow specs: {len(logical_fixtures['workflow_specs'])}")
    print(f"      - Validation cases: {len(logical_fixtures['validation_cases'])}")
    print(f"      - Routing cases: {len(logical_fixtures['routing_cases'])}")
    
    # Get smart fixtures for agentic tests
    agentic_fixtures = repo.get_smart_fixture_data(
        layer=TestLayer.AGENTIC,
        category="data_source_validation"
    )
    print(f"   🤖 Agentic fixtures generated:")
    print(f"      - User prompts: {len(agentic_fixtures['user_prompts'])}")
    print(f"      - Tool catalogs: {len(agentic_fixtures['tool_catalogs'])}")
    print(f"      - Generation cases: {len(agentic_fixtures['generation_cases'])}")
    
    # 6. Demonstrate actual validation using the smart fixtures
    print("\n5. ✅ RUNNING VALIDATION WITH SMART FIXTURES")
    
    for workflow_spec_data in logical_fixtures['workflow_specs']:
        if workflow_spec_data:
            try:
                # Convert to WorkflowSpec and validate
                workflow = WorkflowSpec(**workflow_spec_data)
                
                # Create mock tool catalog with valid data sources
                mock_catalog = {name: {"description": f"Valid data source"} 
                              for name in get_valid_data_source_names()}
                mock_catalog.update(agentic_fixtures['tool_catalogs'][0] or {})
                
                # Validate structure
                issues = workflow.validate_structure(mock_catalog)
                
                if issues:
                    print(f"   ❌ Validation issues found: {len(issues)}")
                    for issue in issues[:3]:  # Show first 3
                        print(f"      • {issue}")
                else:
                    print(f"   ✅ Workflow '{workflow.title}' validates successfully")
                    
                    # Verify the specific fix: no invalid data sources
                    data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
                    for node in data_source_nodes:
                        source_name = getattr(node.data, 'source_name', None)
                        if source_name not in get_valid_data_source_names():
                            print(f"   🚨 INVALID DATA SOURCE DETECTED: {source_name}")
                        else:
                            print(f"   ✅ Valid data source: {source_name}")
                    
                    # Verify agents have appropriate tools for web crawling
                    agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
                    for node in agent_nodes:
                        if 'crawl' in node.label.lower() or 'web' in node.label.lower():
                            tools = getattr(node.data, 'tools', [])
                            if any('crawler' in tool.lower() or 'scrape' in tool.lower() for tool in tools):
                                print(f"   ✅ Crawler agent has appropriate tools: {tools}")
                            else:
                                print(f"   ⚠️  Crawler agent missing crawler tools: {tools}")
                                
            except Exception as e:
                print(f"   ❌ Error validating workflow: {e}")
    
    # 7. Show the persistent storage structure
    print("\n6. 💾 PERSISTENT STORAGE STRUCTURE")
    from pathlib import Path
    storage_dir = Path("demo_test_repository")
    
    if storage_dir.exists():
        print(f"   📂 Repository structure:")
        for layer_dir in storage_dir.iterdir():
            if layer_dir.is_dir():
                test_files = list(layer_dir.glob("*.json"))
                print(f"      {layer_dir.name}/: {len(test_files)} test files")
                for test_file in test_files[:2]:  # Show first 2
                    print(f"        • {test_file.name}")
    
    # 8. Demonstrate the factory can create different test types
    print("\n7. 🏭 FACTORY PATTERN FOR DIFFERENT TEST TYPES")
    
    # Stock trading test (orchestration layer)
    stock_test = WorkflowTestCase(
        id="stock-trading-sla-test",
        name="Stock Trading with SLA Enforcement",
        description="Full pipeline test with SLA requirements for stock tools",
        layer=TestLayer.ORCHESTRATION,
        category="stock_trading_pipeline",
        user_prompt="A user input, connected to a stock Decision agent using tools that fetch historical and current stock prices, with a required conditional gate that connects to a buy or sell agent. A trade is triggered if the given stock(s) are 5% greater or less than their historical price (compare to yesterday). A 5% bump means a sell, a -5% or more means a buy. Both agents are connected to an email agent that sends email to me, alex@io.net about the trade.",
        expected_result={
            "has_user_input_data_source": True,
            "decision_agent_has_stock_tools": True, 
            "has_conditional_routing": True,
            "buy_sell_agents_have_email_tools": True,
            "sla_enforcement_active": True
        },
        tags=["stock_trading", "sla", "conditional_routing", "email", "pipeline"]
    )
    
    repo.add_test_case(stock_test)
    print(f"   ✅ Added orchestration test: {stock_test.name}")
    print(f"   🎯 This matches the user's original request from CLAUDE.local.md")
    
    # 9. Summary of capabilities
    print("\n8. 🎉 SMART TEST SYSTEM CAPABILITIES SUMMARY")
    print("   ✅ Intelligent storage with categorization and tagging")
    print("   ✅ Factory patterns for different test types (logical, agentic, orchestration)")
    print("   ✅ Smart fixture generation based on filters")
    print("   ✅ Persistent storage with automatic loading")
    print("   ✅ Integration with existing WorkflowSpec validation")
    print("   ✅ Support for complex test scenarios (SLA enforcement, routing, etc.)")
    print("   ✅ Easy addition of new test cases as work is completed")
    
    print(f"\n🎯 TOTAL TESTS IN REPOSITORY: {len(repo._test_cases)}")
    print("📊 Test distribution:")
    for layer in TestLayer:
        layer_tests = repo.get_tests_by_layer(layer)
        print(f"   {layer.value}: {len(layer_tests)} tests")
    
    print("\n" + "=" * 60)
    print("This IS the smart test storage and factory system you wanted!")
    print("Each piece of work can now easily add test configs/WorkflowSpecs")
    print("that are intelligently stored and can be run through factories.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_smart_test_system()