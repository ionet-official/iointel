"""
Centralized Workflow Test Repository
====================================

This module provides a centralized system for managing workflow test cases across
different testing layers:

1. Logical Tests - Pure data structure validation (conditional gates, routing)
2. Agentic Tests - LLM-based workflow generation from user prompts  
3. Orchestration Tests - Full pipeline DAG execution and SLA enforcement
4. Feedback Tests - Post-execution chat feedback loops

The repository provides smart fixtures that can supply appropriate test data
for different test scenarios, eliminating the need for scattered test patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from ..web.workflow_storage import WorkflowStorage


class TestLayer(Enum):
    """Different layers of workflow testing."""
    LOGICAL = "logical"          # Pure data structure tests
    AGENTIC = "agentic"         # LLM generation tests  
    ORCHESTRATION = "orchestration"  # Full pipeline tests
    FEEDBACK = "feedback"       # Chat feedback tests


@dataclass
class WorkflowTestCase:
    """A single test case with metadata. Extensible to any test type."""
    
    id: str
    name: str
    description: str
    layer: TestLayer
    category: str  # e.g., "conditional_routing", "stock_analysis", "validation"
    
    # Test type and inputs
    test_type: str = "workflow_validation"  # Type of test executor to use
    user_prompt: Optional[str] = None  # For agentic tests
    workflow_spec: Optional[Dict[str, Any]] = None  # For logical workflow tests
    tool_catalog: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    
    # Expected outcomes
    expected_result: Optional[Dict[str, Any]] = None
    should_pass: bool = True
    expected_errors: Optional[List[str]] = None
    
    # Metadata
    tags: List[str] = None
    created_at: str = None
    updated_at: str = None
    author: str = "system"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass 
class TestSuite:
    """A collection of related test cases."""
    
    id: str
    name: str
    description: str
    layer: TestLayer
    test_cases: List[WorkflowTestCase]
    tags: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class WorkflowTestRepository:
    """Centralized repository for workflow test cases."""
    
    def __init__(self, storage_dir: str = "workflow_test_repository"):
        """
        Initialize the test repository.
        
        Args:
            storage_dir: Directory to store test cases
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create layer-specific directories
        for layer in TestLayer:
            (self.storage_dir / layer.value).mkdir(exist_ok=True)
        
        # Initialize workflow storage for integration
        self.workflow_storage = WorkflowStorage(str(self.storage_dir / "workflows"))
        
        # Load existing test cases
        self._test_cases: Dict[str, WorkflowTestCase] = {}
        self._test_suites: Dict[str, TestSuite] = {}
        self._load_existing_tests()
    
    def _load_existing_tests(self):
        """Load existing test cases from storage."""
        for layer_dir in self.storage_dir.iterdir():
            if layer_dir.is_dir() and layer_dir.name in [l.value for l in TestLayer]:
                for test_file in layer_dir.glob("*.json"):
                    try:
                        with open(test_file) as f:
                            data = json.load(f)
                        
                        if data.get('type') == 'test_case':
                            test_case = WorkflowTestCase(**data['data'])
                            test_case.layer = TestLayer(test_case.layer.value if isinstance(test_case.layer, TestLayer) else test_case.layer)
                            self._test_cases[test_case.id] = test_case
                        elif data.get('type') == 'test_suite':
                            test_suite_data = data['data']
                            test_cases = [WorkflowTestCase(**tc) for tc in test_suite_data.pop('test_cases', [])]
                            test_suite = TestSuite(**test_suite_data, test_cases=test_cases)
                            test_suite.layer = TestLayer(test_suite.layer.value if isinstance(test_suite.layer, TestLayer) else test_suite.layer)
                            self._test_suites[test_suite.id] = test_suite
                    except Exception as e:
                        print(f"Failed to load test from {test_file}: {e}")
    
    def add_test_case(self, test_case: WorkflowTestCase) -> str:
        """
        Add a new test case to the repository.
        
        Args:
            test_case: The test case to add
            
        Returns:
            The test case ID
        """
        test_case.updated_at = datetime.now().isoformat()
        self._test_cases[test_case.id] = test_case
        self._save_test_case(test_case)
        return test_case.id
    
    def add_test_suite(self, test_suite: TestSuite) -> str:
        """
        Add a new test suite to the repository.
        
        Args:
            test_suite: The test suite to add
            
        Returns:
            The test suite ID
        """
        self._test_suites[test_suite.id] = test_suite
        self._save_test_suite(test_suite)
        return test_suite.id
    
    def _save_test_case(self, test_case: WorkflowTestCase):
        """Save a test case to disk."""
        file_path = self.storage_dir / test_case.layer.value / f"{test_case.id}.json"
        
        # Convert dataclass to dict, handling enum serialization
        test_data = asdict(test_case)
        test_data['layer'] = test_case.layer.value
        
        with open(file_path, 'w') as f:
            json.dump({
                'type': 'test_case',
                'data': test_data
            }, f, indent=2)
    
    def _save_test_suite(self, test_suite: TestSuite):
        """Save a test suite to disk."""
        file_path = self.storage_dir / test_suite.layer.value / f"suite_{test_suite.id}.json"
        
        # Convert dataclass to dict, handling enum serialization
        suite_data = asdict(test_suite)
        suite_data['layer'] = test_suite.layer.value
        
        with open(file_path, 'w') as f:
            json.dump({
                'type': 'test_suite', 
                'data': suite_data
            }, f, indent=2)
    
    def get_test_case(self, test_id: str) -> Optional[WorkflowTestCase]:
        """Get a test case by ID."""
        return self._test_cases.get(test_id)
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        return self._test_suites.get(suite_id)
    
    def get_tests_by_layer(self, layer: TestLayer) -> List[WorkflowTestCase]:
        """Get all test cases for a specific layer."""
        return [tc for tc in self._test_cases.values() if tc.layer == layer]
    
    def get_tests_by_category(self, category: str) -> List[WorkflowTestCase]:
        """Get all test cases for a specific category."""
        return [tc for tc in self._test_cases.values() if tc.category == category]
    
    def get_tests_by_tags(self, tags: List[str]) -> List[WorkflowTestCase]:
        """Get all test cases that have any of the specified tags."""
        return [tc for tc in self._test_cases.values() 
                if any(tag in tc.tags for tag in tags)]
    
    def get_all_tests(self) -> List[WorkflowTestCase]:
        """Get all test cases in the repository."""
        return list(self._test_cases.values())
    
    def create_logical_test(
        self,
        name: str,
        description: str,
        category: str,
        workflow_spec: Dict[str, Any],
        expected_result: Optional[Dict[str, Any]] = None,
        should_pass: bool = True,
        expected_errors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        test_type: str = "workflow_validation"
    ) -> WorkflowTestCase:
        """
        Create a logical layer test case.
        
        These test pure data structures - conditional gates, validation, routing logic.
        No LLM involvement. Can specify custom test_type for different executors.
        """
        test_case = WorkflowTestCase(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            layer=TestLayer.LOGICAL,
            category=category,
            test_type=test_type,
            workflow_spec=workflow_spec,
            expected_result=expected_result,
            should_pass=should_pass,
            expected_errors=expected_errors,
            tags=tags or []
        )
        self.add_test_case(test_case)
        return test_case
    
    def create_agentic_test(
        self,
        name: str,
        description: str,
        category: str,
        user_prompt: str,
        tool_catalog: Optional[Dict[str, Any]] = None,
        expected_result: Optional[Dict[str, Any]] = None,
        should_pass: bool = True,
        expected_errors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> WorkflowTestCase:
        """
        Create an agentic layer test case.
        
        These test LLM-based workflow generation from user prompts.
        """
        test_case = WorkflowTestCase(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            layer=TestLayer.AGENTIC,
            category=category,
            user_prompt=user_prompt,
            tool_catalog=tool_catalog,
            expected_result=expected_result,
            should_pass=should_pass,
            expected_errors=expected_errors,
            tags=tags or []
        )
        self.add_test_case(test_case)
        return test_case
    
    def create_orchestration_test(
        self,
        name: str,
        description: str,
        category: str,
        user_prompt: str,
        expected_result: Optional[Dict[str, Any]] = None,
        should_pass: bool = True,
        expected_errors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowTestCase:
        """
        Create an orchestration layer test case.
        
        These test full workflow execution with DAG execution and SLA enforcement.
        """
        test_case = WorkflowTestCase(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            layer=TestLayer.ORCHESTRATION,
            category=category,
            test_type="workflow_execution",
            user_prompt=user_prompt,
            expected_result=expected_result,
            should_pass=should_pass,
            expected_errors=expected_errors,
            tags=tags or [],
            context=execution_config or {}
        )
        self.add_test_case(test_case)
        return test_case
    
    def scan_existing_tests(self, test_directories: List[Path]) -> Dict[str, List[WorkflowTestCase]]:
        """
        Scan existing test directories and extract workflow test cases.
        
        This method analyzes existing test files to identify distinct test patterns
        and converts them into the centralized format.
        
        Args:
            test_directories: List of directories to scan for tests
            
        Returns:
            Dictionary mapping test file paths to extracted test cases
        """
        extracted_tests = {}
        
        for test_dir in test_directories:
            if not test_dir.exists():
                continue
                
            for test_file in test_dir.rglob("test_*.py"):
                try:
                    # Read and parse test file
                    content = test_file.read_text()
                    test_cases = self._extract_test_cases_from_file(content, test_file)
                    if test_cases:
                        extracted_tests[str(test_file)] = test_cases
                except Exception as e:
                    print(f"Failed to scan {test_file}: {e}")
        
        return extracted_tests
    
    def _extract_test_cases_from_file(self, content: str, file_path: Path) -> List[WorkflowTestCase]:
        """
        Extract workflow test cases from a Python test file.
        
        This is a basic implementation that looks for common patterns.
        Could be enhanced with AST parsing for more sophisticated extraction.
        """
        test_cases = []
        
        # Look for workflow generation patterns
        if "generate_workflow" in content and "query" in content:
            # This appears to be an agentic test
            test_case = WorkflowTestCase(
                id=str(uuid.uuid4()),
                name=f"Extracted from {file_path.name}",
                description=f"Agentic test extracted from {file_path}",
                layer=TestLayer.AGENTIC,
                category="extracted",
                tags=["extracted", "legacy"]
            )
            test_cases.append(test_case)
        
        # Look for conditional routing patterns
        if "conditional" in content.lower() and "routing" in content.lower():
            test_case = WorkflowTestCase(
                id=str(uuid.uuid4()),
                name=f"Conditional routing from {file_path.name}",
                description=f"Logical test extracted from {file_path}",
                layer=TestLayer.LOGICAL,
                category="conditional_routing",
                tags=["extracted", "legacy", "routing"]
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def get_smart_fixture_data(
        self, 
        layer: TestLayer, 
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get smart fixture data for tests based on layer and category.
        
        This provides appropriate test data for different testing scenarios,
        eliminating the need for scattered test patterns.
        
        Args:
            layer: The test layer to get fixtures for
            category: Optional category filter
            tags: Optional tag filters
            
        Returns:
            Dictionary containing appropriate fixture data
        """
        # Get relevant test cases
        test_cases = self.get_tests_by_layer(layer)
        
        if category:
            test_cases = [tc for tc in test_cases if tc.category == category]
        
        if tags:
            test_cases = [tc for tc in test_cases if any(tag in tc.tags for tag in tags)]
        
        # Build fixture data based on layer
        if layer == TestLayer.LOGICAL:
            return self._build_logical_fixtures(test_cases)
        elif layer == TestLayer.AGENTIC:
            return self._build_agentic_fixtures(test_cases)
        elif layer == TestLayer.ORCHESTRATION:
            return self._build_orchestration_fixtures(test_cases)
        elif layer == TestLayer.FEEDBACK:
            return self._build_feedback_fixtures(test_cases)
        
        return {}
    
    def _build_logical_fixtures(self, test_cases: List[WorkflowTestCase]) -> Dict[str, Any]:
        """Build fixture data for logical layer tests."""
        return {
            'workflow_specs': [tc.workflow_spec for tc in test_cases if tc.workflow_spec],
            'validation_cases': [tc for tc in test_cases if tc.expected_errors],
            'routing_cases': [tc for tc in test_cases if 'routing' in tc.category],
        }
    
    def _build_agentic_fixtures(self, test_cases: List[WorkflowTestCase]) -> Dict[str, Any]:
        """Build fixture data for agentic layer tests."""
        return {
            'user_prompts': [tc.user_prompt for tc in test_cases if tc.user_prompt],
            'tool_catalogs': [tc.tool_catalog for tc in test_cases if tc.tool_catalog],
            'generation_cases': test_cases,
        }
    
    def _build_orchestration_fixtures(self, test_cases: List[WorkflowTestCase]) -> Dict[str, Any]:
        """Build fixture data for orchestration layer tests."""
        return {
            'execution_cases': test_cases,
            'sla_enforcement_cases': [tc for tc in test_cases if 'sla' in tc.tags],
            'pipeline_cases': [tc for tc in test_cases if 'pipeline' in tc.tags],
        }
    
    def _build_feedback_fixtures(self, test_cases: List[WorkflowTestCase]) -> Dict[str, Any]:
        """Build fixture data for feedback layer tests."""
        return {
            'chat_cases': test_cases,
            'feedback_loops': [tc for tc in test_cases if 'feedback' in tc.tags],
        }
    
    def create_default_test_cases(self):
        """Create a set of default test cases for each layer."""
        
        # Logical layer defaults
        self.create_logical_test(
            name="Basic conditional routing",
            description="Test conditional routing with buy/sell decision",
            category="conditional_routing",
            workflow_spec={
                "nodes": [
                    {"id": "decision", "type": "decision", "label": "Buy/Sell Decision"},
                    {"id": "buy_agent", "type": "agent", "label": "Buy Agent"},
                    {"id": "sell_agent", "type": "agent", "label": "Sell Agent"}
                ],
                "edges": [
                    {"source": "decision", "target": "buy_agent", "data": {"route_index": 0}},
                    {"source": "decision", "target": "sell_agent", "data": {"route_index": 1}}
                ]
            },
            tags=["routing", "decision", "stock_trading"]
        )
        
        # Agentic layer defaults
        self.create_agentic_test(
            name="Stock agent generation",
            description="Test generating a stock analysis workflow from user prompt",
            category="stock_analysis",
            user_prompt="stock agent",
            tool_catalog={
                "user_input": {"description": "Get user input"},
                "yfinance.get_stock_info": {"description": "Get stock information"},
                "get_current_stock_price": {"description": "Get current stock price"}
            },
            expected_result={
                "has_data_source_nodes": True,
                "has_agent_nodes": True,
                "agent_nodes_have_stock_tools": True
            },
            tags=["stock", "generation", "data_source_fix"]
        )
        
        # Orchestration layer defaults  
        test_case = WorkflowTestCase(
            id=str(uuid.uuid4()),
            name="Full stock trading pipeline",
            description="End-to-end test of stock trading workflow execution",
            layer=TestLayer.ORCHESTRATION,
            category="stock_trading_pipeline",
            user_prompt="A user input, connected to a stock Decision agent using tools that fetch historical and current stock prices, with a required conditional gate that connects to a buy or sell agent. A trade is triggered if the given stock(s) are 5% greater or less than their historical price. A 5% bump means a sell, a -5% or more means a buy. Both agents are connected to an email agent that sends email to me, alex@io.net about the trade.",
            expected_result={
                "executes_successfully": True,
                "sends_email": True,
                "follows_sla_requirements": True
            },
            tags=["stock_trading", "pipeline", "email", "sla", "execution"]
        )
        self.add_test_case(test_case)
        
        print("✅ Created default test cases for all layers")


# Global test repository instance
_test_repository = None

def get_test_repository() -> WorkflowTestRepository:
    """Get the global test repository instance."""
    global _test_repository
    if _test_repository is None:
        _test_repository = WorkflowTestRepository()
    return _test_repository