#!/usr/bin/env python3
"""
Proper Usage of Smart Test Repository System
============================================

This demonstrates the correct way to add our data source validation test
to the smart repository system, showing how it actually IS the intelligent
storage and factory system the user expected.
"""

from uuid import uuid4
from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository, 
    TestLayer, 
    WorkflowTestCase
)
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.data_models.data_source_registry import get_valid_data_source_names


def add_data_source_validation_test_properly():
    """Add our data source validation test using the smart system."""
    print("🧪 ADDING DATA SOURCE VALIDATION TEST TO SMART REPOSITORY")
    print("=" * 60)
    
    # Initialize the repository
    repo = WorkflowTestRepository(storage_dir="smart_test_repository")
    
    # Create the test case using the smart factory with proper UUID
    test_case = repo.create_logical_test(
        name="Data Source Registry Validation",
        description="Validates that WorkflowPlanner only uses valid data source names and prevents 'crawl_the_web' type mistakes",
        category="data_source_validation",
        workflow_spec={
            "id": str(uuid4()),  # Proper UUID
            "rev": 1,
            "title": "Data Source Validation Test", 
            "description": "Test that ensures only valid data sources are used",
            "nodes": [
                {
                    "id": "user_input_1",
                    "type": "data_source",
                    "label": "User Input",
                    "data": {
                        "source_name": "user_input",  # Valid from registry
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
                        "agent_instructions": "Crawl the web for information based on user query",
                        "tools": ["Crawler-scrape_url"],  # Correct: agent with tools, not data_source
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
        should_pass=True,
        tags=["data_source", "validation", "crawl_web_fix", "registry", "anti_hallucination"]
    )
    
    print(f"✅ Created and stored test: {test_case.name}")
    print(f"🆔 Test ID: {test_case.id}")
    print(f"🎯 Category: {test_case.category}")
    print(f"🏷️  Tags: {test_case.tags}")
    
    # Validate the workflow spec we just stored
    print("\n🔍 VALIDATING THE STORED WORKFLOW SPEC")
    workflow = WorkflowSpec(**test_case.workflow_spec)
    
    # Create a realistic tool catalog
    mock_catalog = {
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
    }
    
    issues = workflow.validate_structure(mock_catalog)
    
    if issues:
        print(f"❌ Validation issues: {len(issues)}")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Workflow validates successfully!")
        
        # Verify the specific fixes
        data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
        agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
        
        print(f"   📊 Data source nodes: {len(data_source_nodes)}")
        for node in data_source_nodes:
            source_name = node.data.source_name
            if source_name in get_valid_data_source_names():
                print(f"   ✅ Valid data source: {source_name}")
            else:
                print(f"   ❌ Invalid data source: {source_name}")
        
        print(f"   🤖 Agent nodes: {len(agent_nodes)}")
        for node in agent_nodes:
            tools = node.data.tools or []
            print(f"   ✅ Agent '{node.label}' has tools: {tools}")
    
    # Show how to retrieve this test intelligently
    print("\n🔍 INTELLIGENT TEST RETRIEVAL")
    
    # Get by category
    validation_tests = repo.get_tests_by_category("data_source_validation")
    print(f"📊 Tests in data_source_validation category: {len(validation_tests)}")
    
    # Get by tags
    anti_hallucination_tests = repo.get_tests_by_tags(["anti_hallucination"])
    print(f"🏷️  Anti-hallucination tests: {len(anti_hallucination_tests)}")
    
    # Get smart fixtures for this category
    fixtures = repo.get_smart_fixture_data(
        layer=TestLayer.LOGICAL,
        category="data_source_validation"
    )
    print(f"🧪 Generated fixtures:")
    print(f"   - Workflow specs: {len(fixtures['workflow_specs'])}")
    print(f"   - Validation cases: {len(fixtures['validation_cases'])}")
    
    # Show that this is persistent and reloadable
    print("\n💾 PERSISTENCE & RELOADING")
    
    # Create a new repository instance
    repo2 = WorkflowTestRepository(storage_dir="smart_test_repository")
    reloaded_test = repo2.get_test_case(test_case.id)
    
    if reloaded_test:
        print(f"✅ Test successfully reloaded from disk")
        print(f"   Name: {reloaded_test.name}")
        print(f"   Category: {reloaded_test.category}")
        print(f"   Has workflow spec: {reloaded_test.workflow_spec is not None}")
    else:
        print("❌ Failed to reload test")
    
    return test_case


def demonstrate_factory_patterns():
    """Show different factory patterns for different test types."""
    print("\n🏭 FACTORY PATTERNS FOR DIFFERENT TEST TYPES")
    print("=" * 50)
    
    repo = WorkflowTestRepository(storage_dir="smart_test_repository")
    
    # 1. Logical layer factory
    print("1. ⚗️  LOGICAL LAYER FACTORY")
    logical_test = repo.create_logical_test(
        name="SLA Enforcement Structure",
        description="Test that SLA requirements are properly structured in WorkflowSpec",
        category="sla_enforcement",
        workflow_spec={
            "id": str(uuid4()),
            "rev": 1,
            "title": "SLA Test Workflow",
            "description": "Workflow with SLA requirements",
            "nodes": [
                {
                    "id": "decision_1",
                    "type": "decision",
                    "label": "Stock Decision Agent",
                    "data": {
                        "agent_instructions": "Analyze stock and make decision",
                        "tools": ["get_current_stock_price", "conditional_gate"],
                        "ins": ["stock_data"],
                        "outs": ["buy", "sell"]
                    },
                    "sla": {
                        "tool_usage_required": True,
                        "required_tools": ["get_current_stock_price"],
                        "final_tool_must_be": "conditional_gate",
                        "enforce_usage": True
                    }
                }
            ],
            "edges": []
        },
        tags=["sla", "enforcement", "structure"]
    )
    print(f"   ✅ Created logical test: {logical_test.name}")
    
    # 2. Agentic layer factory
    print("2. 🤖 AGENTIC LAYER FACTORY")
    agentic_test = repo.create_agentic_test(
        name="Stock Agent Generation with SLA",
        description="Test that generated stock workflows include proper SLA enforcement",
        category="stock_generation",
        user_prompt="A user input, connected to a stock Decision agent using tools that fetch historical and current stock prices, with a required conditional gate that connects to a buy or sell agent. A trade is triggered if the given stock(s) are 5% greater or less than their historical price (compare to yesterday). A 5% bump means a sell, a -5% or more means a buy. Both agents are connected to an email agent that sends email to me, alex@io.net about the trade.",
        tool_catalog={
            "user_input": {"description": "Get user input"},
            "get_current_stock_price": {"description": "Get current stock price"}, 
            "conditional_gate": {"description": "Route based on conditions"},
            "send_email": {"description": "Send email notifications"}
        },
        expected_result={
            "has_decision_agent_with_stock_tools": True,
            "decision_agent_has_sla_enforcement": True,
            "has_conditional_routing": True,
            "buy_sell_agents_have_email": True
        },
        tags=["stock", "generation", "sla", "conditional_routing", "email"]
    )
    print(f"   ✅ Created agentic test: {agentic_test.name}")
    
    # 3. Orchestration layer (full pipeline test)
    orchestration_test = WorkflowTestCase(
        id=str(uuid4()),
        name="End-to-End Stock Trading Pipeline", 
        description="Full execution test of stock trading workflow with SLA enforcement",
        layer=TestLayer.ORCHESTRATION,
        category="stock_trading_pipeline",
        user_prompt=agentic_test.user_prompt,  # Same prompt as agentic test
        tool_catalog=agentic_test.tool_catalog,  # Same tools
        expected_result={
            "workflow_executes_successfully": True,
            "sla_enforcement_applied": True,
            "conditional_routing_works": True,
            "emails_sent_successfully": True,
            "stock_prices_fetched": True
        },
        tags=["pipeline", "execution", "stock_trading", "sla", "integration"]
    )
    repo.add_test_case(orchestration_test)
    print(f"3. 🎯 ORCHESTRATION TEST: {orchestration_test.name}")
    
    # Show the smart relationship between test layers
    print("\n🔗 SMART TEST RELATIONSHIPS")
    print("   The same user prompt flows through all layers:")
    print(f"   🤖 Agentic: Tests generation from prompt")
    print(f"   ⚗️  Logical: Tests resulting structure")
    print(f"   🎯 Orchestration: Tests full execution")
    
    return logical_test, agentic_test, orchestration_test


def show_repository_state():
    """Show the current state of the repository."""
    print("\n📊 SMART REPOSITORY STATE")
    print("=" * 30)
    
    repo = WorkflowTestRepository(storage_dir="smart_test_repository")
    
    print(f"🎯 Total tests: {len(repo._test_cases)}")
    
    # Show distribution by layer
    for layer in TestLayer:
        layer_tests = repo.get_tests_by_layer(layer)
        print(f"   {layer.value}: {len(layer_tests)} tests")
        for test in layer_tests:
            print(f"      • {test.name} ({test.category})")
    
    # Show categories
    categories = set(test.category for test in repo._test_cases.values())
    print(f"\n📁 Categories: {len(categories)}")
    for category in sorted(categories):
        count = len(repo.get_tests_by_category(category))
        print(f"   {category}: {count} tests")
    
    # Show tags
    all_tags = set()
    for test in repo._test_cases.values():
        all_tags.update(test.tags)
    print(f"\n🏷️  Tags: {len(all_tags)}")
    for tag in sorted(all_tags):
        count = len(repo.get_tests_by_tags([tag]))
        print(f"   {tag}: {count} tests")


if __name__ == "__main__":
    # Add our specific data source validation test
    test_case = add_data_source_validation_test_properly()
    
    # Show different factory patterns
    logical, agentic, orchestration = demonstrate_factory_patterns()
    
    # Show final repository state
    show_repository_state()
    
    print("\n" + "=" * 60)
    print("🎉 THIS IS THE SMART TEST SYSTEM YOU WANTED!")
    print("=" * 60)
    print("✅ Intelligent storage with categorization and tagging")
    print("✅ Factory patterns for creating different test types") 
    print("✅ Smart fixtures based on filters and categories")
    print("✅ Persistent storage that auto-loads on restart")
    print("✅ Easy addition of new tests as work is completed")
    print("✅ Integration with existing WorkflowSpec validation")
    print("✅ Support for all test layers (logical, agentic, orchestration)")
    print("\n💡 Each piece of successful work can now add test configs")
    print("   and WorkflowSpecs that are intelligently managed!")
    print("=" * 60)