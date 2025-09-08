"""
Smart Workflow Test Fixtures
============================

This module provides intelligent pytest fixtures that can supply appropriate
test data for different workflow testing layers and scenarios.

Instead of each test reinventing its own data structures, tests can simply
request fixtures like:
- logical_workflow_data
- agentic_test_prompts  
- orchestration_pipeline_data
- feedback_chat_data

The fixtures automatically provide relevant test cases from the centralized
test repository based on the test's needs.
"""

import pytest
from pathlib import Path

from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository, 
    TestLayer, 
    WorkflowTestCase,
    get_test_repository
)
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env


@pytest.fixture(scope="session")
def test_repository() -> WorkflowTestRepository:
    """Global test repository instance for the test session."""
    repo = get_test_repository()
    
    # Initialize with default test cases if empty
    if not repo._test_cases:
        repo.create_default_test_cases()
    
    return repo


@pytest.fixture(scope="session") 
def real_tool_catalog():
    """Real tool catalog loaded from environment for integration tests."""
    # Load tools from environment first (this registers them)
    load_tools_from_env('creds.env')
    
    # Create and return the real tool catalog
    return create_tool_catalog()


@pytest.fixture
def mock_tool_catalog():
    """Mock tool catalog for unit tests that don't need real tools."""
    return {
        "user_input": {
            "name": "user_input",
            "description": "Get input from user",
            "parameters": {"prompt": {"type": "str", "description": "Prompt to show user"}},
            "required_parameters": ["prompt"]
        },
        "weather_api": {
            "name": "weather_api", 
            "description": "Get weather information for a location",
            "parameters": {
                "location": {"type": "str", "description": "Location to get weather for"},
                "units": {"type": "str", "description": "Temperature units", "default": "celsius"}
            },
            "required_parameters": ["location"]
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {
                "to": {"type": "str", "description": "Recipient email address"},
                "subject": {"type": "str", "description": "Email subject"},
                "body": {"type": "str", "description": "Email body"}
            },
            "required_parameters": ["to", "subject", "body"]
        },
        "get_current_stock_price": {
            "name": "get_current_stock_price",
            "description": "Get current stock price for a symbol",
            "parameters": {
                "symbol": {"type": "str", "description": "Stock symbol (e.g., AAPL)"},
                "exchange": {"type": "str", "description": "Stock exchange", "default": "NASDAQ"}
            },
            "required_parameters": ["symbol"]
        },
        "yfinance.get_stock_info": {
            "name": "yfinance.get_stock_info",
            "description": "Get detailed stock information using yfinance",
            "parameters": {
                "symbol": {"type": "str", "description": "Stock symbol"},
                "period": {"type": "str", "description": "Time period", "default": "1d"}
            },
            "required_parameters": ["symbol"]
        }
    }


# LOGICAL LAYER FIXTURES
@pytest.fixture
def logical_workflow_data(test_repository):
    """Get workflow data for logical layer tests (pure data structure tests)."""
    return test_repository.get_smart_fixture_data(TestLayer.LOGICAL)


@pytest.fixture
def conditional_routing_cases(test_repository):
    """Get test cases specifically for conditional routing logic."""
    return test_repository.get_tests_by_category("conditional_routing")


@pytest.fixture
def validation_test_cases(test_repository):
    """Get test cases for workflow validation logic."""
    logical_data = test_repository.get_smart_fixture_data(TestLayer.LOGICAL)
    return logical_data.get('validation_cases', [])


# AGENTIC LAYER FIXTURES  
@pytest.fixture
def agentic_test_data(test_repository):
    """Get test data for agentic layer tests (LLM-based workflow generation)."""
    return test_repository.get_smart_fixture_data(TestLayer.AGENTIC)


@pytest.fixture
def stock_analysis_prompts(test_repository):
    """Get user prompts for stock analysis workflow generation."""
    stock_tests = test_repository.get_tests_by_category("stock_analysis")
    return [test.user_prompt for test in stock_tests if test.user_prompt]


@pytest.fixture 
def workflow_generation_cases(test_repository):
    """Get test cases for workflow generation from user prompts."""
    agentic_data = test_repository.get_smart_fixture_data(TestLayer.AGENTIC)
    return agentic_data.get('generation_cases', [])


# ORCHESTRATION LAYER FIXTURES
@pytest.fixture
def orchestration_test_data(test_repository):
    """Get test data for orchestration layer tests (full pipeline execution)."""
    return test_repository.get_smart_fixture_data(TestLayer.ORCHESTRATION)


@pytest.fixture
def pipeline_execution_cases(test_repository):
    """Get test cases for full pipeline execution."""
    return test_repository.get_tests_by_tags(["pipeline", "execution"])


@pytest.fixture
def sla_enforcement_cases(test_repository):
    """Get test cases for SLA enforcement testing."""
    return test_repository.get_tests_by_tags(["sla"])


# FEEDBACK LAYER FIXTURES
@pytest.fixture
def feedback_test_data(test_repository):
    """Get test data for feedback layer tests (chat feedback loops)."""
    return test_repository.get_smart_fixture_data(TestLayer.FEEDBACK)


@pytest.fixture
def chat_feedback_cases(test_repository):
    """Get test cases for chat feedback loops."""
    feedback_data = test_repository.get_smart_fixture_data(TestLayer.FEEDBACK)
    return feedback_data.get('feedback_loops', [])


# CATEGORY-SPECIFIC FIXTURES
@pytest.fixture
def stock_trading_tests(test_repository):
    """Get all test cases related to stock trading across all layers."""
    return test_repository.get_tests_by_tags(["stock_trading", "stock"])


@pytest.fixture
def data_source_vs_agent_tests(test_repository):
    """Get test cases for the data_source vs agent node distinction fix."""
    return test_repository.get_tests_by_tags(["data_source_fix"])


# SMART FIXTURE DISPATCHER
@pytest.fixture
def smart_test_data(request, test_repository):
    """
    Smart fixture that provides appropriate test data based on test markers.
    
    Usage:
        @pytest.mark.layer("logical")
        @pytest.mark.category("conditional_routing") 
        def test_something(smart_test_data):
            # Gets logical layer conditional routing test data automatically
    """
    # Get markers from the test
    layer_marker = request.node.get_closest_marker("layer")
    category_marker = request.node.get_closest_marker("category")
    tags_marker = request.node.get_closest_marker("tags")
    
    # Default to logical layer if not specified
    layer = TestLayer.LOGICAL
    if layer_marker:
        layer = TestLayer(layer_marker.args[0])
    
    category = None
    if category_marker:
        category = category_marker.args[0]
    
    tags = []
    if tags_marker:
        tags = list(tags_marker.args)
    
    return test_repository.get_smart_fixture_data(layer, category, tags)


# SCANNING AND EXTRACTION FIXTURES
@pytest.fixture(scope="session")
def scanned_legacy_tests(test_repository):
    """
    Scan existing test directories and extract legacy test cases.
    
    This fixture runs once per session and extracts test patterns from
    existing test files to populate the centralized repository.
    """
    test_dirs = [
        Path("tests/workflows"),
        Path("tests/routing"), 
        Path("tests/tools"),
        Path("tests/integrations")
    ]
    
    return test_repository.scan_existing_tests(test_dirs)


# UTILITY FIXTURES
@pytest.fixture
def workflow_test_case_factory(test_repository):
    """Factory for creating new test cases during tests."""
    def create_test_case(layer: TestLayer, category: str, **kwargs) -> WorkflowTestCase:
        if layer == TestLayer.LOGICAL:
            return test_repository.create_logical_test(
                name=kwargs.get('name', f'Test {category}'),
                description=kwargs.get('description', f'Test for {category}'),
                category=category,
                workflow_spec=kwargs.get('workflow_spec', {}),
                **{k: v for k, v in kwargs.items() if k not in ['name', 'description', 'workflow_spec']}
            )
        elif layer == TestLayer.AGENTIC:
            return test_repository.create_agentic_test(
                name=kwargs.get('name', f'Test {category}'),
                description=kwargs.get('description', f'Test for {category}'),
                category=category,
                user_prompt=kwargs.get('user_prompt', 'test prompt'),
                **{k: v for k, v in kwargs.items() if k not in ['name', 'description', 'user_prompt']}
            )
        else:
            # For other layers, create generic test case
            from iointel.src.utilities.workflow_test_repository import WorkflowTestCase
            import uuid
            return WorkflowTestCase(
                id=str(uuid.uuid4()),
                name=kwargs.get('name', f'Test {category}'),
                description=kwargs.get('description', f'Test for {category}'),
                layer=layer,
                category=category,
                **{k: v for k, v in kwargs.items() if k not in ['name', 'description']}
            )
    
    return create_test_case


# BACKWARD COMPATIBILITY FIXTURES
@pytest.fixture
def tool_catalog(mock_tool_catalog):
    """Backward compatibility fixture for existing tests."""
    return mock_tool_catalog


@pytest.fixture
def sample_workflow_spec():
    """Backward compatibility fixture for existing tests."""
    import uuid
    from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
    
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Sample Test Workflow",
        description="A sample workflow for testing",
        nodes=[
            NodeSpec(
                id="input",
                type="data_source", 
                label="User Input",
                data=NodeData(
                    source_name="user_input",
                    config={"prompt": "Enter your request"},
                    ins=[],
                    outs=["user_query"]
                )
            ),
            NodeSpec(
                id="processor", 
                type="agent",
                label="Process Request",
                data=NodeData(
                    agent_instructions="Process the user request and provide a response",
                    tools=["weather_api"],
                    ins=["user_query"], 
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="input_to_processor",
                source="input",
                target="processor", 
                sourceHandle="user_query",
                targetHandle="user_query",
                data=EdgeData()
            )
        ]
    )