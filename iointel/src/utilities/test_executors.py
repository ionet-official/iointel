"""
Test Executor Architecture
==========================

Pluggable test executors for different test types in the unified test system.
This allows extending the test framework to support ANY kind of test - not just workflows.

Architecture:
- Base TestExecutor interface
- Concrete executors for different test types
- Registry system for plugging in new executors
- Factory pattern using Pydantic models for extensible test handlers
- Backward compatibility with existing workflow tests
"""

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, Optional, Callable, List
import importlib
import inspect
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


class TestHandlerSpec(BaseModel):
    """Specification for a test handler function."""
    name: str = Field(..., description="Human-readable name for this test handler")
    keywords: List[str] = Field(..., description="Keywords that trigger this handler")
    handler_func: str = Field(..., description="Name of the method to call")
    description: str = Field("", description="Description of what this handler does")
    
    class Config:
        arbitrary_types_allowed = True


class TestExecutorFactory:
    """Factory for creating test handlers using Pydantic specifications."""
    
    def __init__(self):
        self.handlers: Dict[str, TestHandlerSpec] = {}
        
    def register_handler(self, spec: TestHandlerSpec) -> None:
        """Register a test handler specification."""
        self.handlers[spec.name] = spec
        
    def find_handler(self, test_name: str) -> Optional[TestHandlerSpec]:
        """Find the best matching handler for a test name."""
        test_name_lower = test_name.lower()
        
        # Score each handler based on keyword matches
        best_match = None
        best_score = 0
        
        for handler in self.handlers.values():
            score = 0
            for keyword in handler.keywords:
                if keyword.lower() in test_name_lower:
                    score += 1
            
            # Require at least one keyword match
            if score > 0 and score > best_score:
                best_score = score
                best_match = handler
                
        return best_match
    
    def get_all_handlers(self) -> List[TestHandlerSpec]:
        """Get all registered handlers."""
        return list(self.handlers.values())


class TestExecutor(ABC):
    """Base class for test executors."""
    
    @abstractmethod
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test case and return results.
        
        Args:
            test_case: The test case to execute
            context: Test context (tool_catalog, etc.)
            
        Returns:
            Dict with 'success': bool and other result data
        """
        pass
    
    @abstractmethod
    def can_handle(self, test_type: str) -> bool:
        """Check if this executor can handle the given test type."""
        pass


class WorkflowValidationExecutor(TestExecutor):
    """Executor for workflow validation tests (existing behavior)."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow validation test."""
        from ..agent_methods.data_models.workflow_spec import WorkflowSpec
        
        try:
            # Convert dict to WorkflowSpec if needed
            if isinstance(test_case.workflow_spec, dict):
                spec = WorkflowSpec.model_validate(test_case.workflow_spec)
            else:
                spec = test_case.workflow_spec
            
            # Validate structure
            tool_catalog = context.get('tool_catalog', {})
            issues = spec.validate_structure(tool_catalog)
            
            success = (not issues) if test_case.should_pass else bool(issues)
            
            return {
                'success': success,
                'validation_issues': issues,
                'spec_generated': True,
                'llm_prompt_length': len(spec.to_llm_prompt()) if hasattr(spec, 'to_llm_prompt') else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'validation_issues': [str(e)]
            }
    
    def can_handle(self, test_type: str) -> bool:
        """Handle workflow validation tests."""
        return test_type == "workflow_validation"


class PythonFunctionExecutor(TestExecutor):
    """Executor for Python function-based tests (like semantic RAG tests)."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python function test."""
        try:
            # Get test function name from workflow_spec
            if not test_case.workflow_spec or 'test_function' not in test_case.workflow_spec:
                return {
                    'success': False,
                    'error': 'No test_function specified in workflow_spec'
                }
            
            function_name = test_case.workflow_spec['test_function']
            
            # Try to import and execute the function
            # First try from the test module that created this test
            test_function = self._find_test_function(function_name)
            
            if not test_function:
                return {
                    'success': False,
                    'error': f'Test function {function_name} not found'
                }
            
            # Execute the test function
            if inspect.iscoroutinefunction(test_function):
                result = await test_function()
            else:
                result = test_function()
            
            # Check if result matches expected
            expected = test_case.expected_result or {}
            success = self._validate_result(result, expected)
            
            return {
                'success': success,
                'result': result,
                'expected': expected,
                'function_executed': function_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'function_executed': test_case.workflow_spec.get('test_function', 'unknown')
            }
    
    def _find_test_function(self, function_name: str):
        """Find test function by name."""
        # Try to find function in test_semantic_rag module
        try:
            import test_semantic_rag
            if hasattr(test_semantic_rag, function_name):
                return getattr(test_semantic_rag, function_name)
        except ImportError:
            pass
        
        # Try to find in global namespace (if run from same process)
        if function_name in globals():
            return globals()[function_name]
            
        return None
    
    def _validate_result(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Validate actual result against expected."""
        print("\n=== Validating Test Results ===")
        print(f"Actual result: {json.dumps(actual, indent=2)}")
        print(f"Expected result: {json.dumps(expected, indent=2)}")

        if not expected:
            print("✓ No expectations specified - any result is valid")
            return True
            
        print("\nChecking each expected key/value pair:")
        for key, expected_value in expected.items():
            print(f"\nChecking key: '{key}'")
            
            if key not in actual:
                print(f"✗ Key '{key}' missing from actual result")
                print("Validation failed - missing expected key")
                return False
                
            print(f"Expected value: {expected_value}")
            print(f"Actual value: {actual[key]}")
            
            if actual[key] != expected_value:
                print(f"✗ Values don't match for key '{key}'")
                print("Validation failed - value mismatch") 
                return False
            else:
                print(f"✓ Values match for key '{key}'")
                
        print("\n✓ All validations passed successfully!")
        return True
    
    def can_handle(self, test_type: str) -> bool:
        """Handle Python function tests."""
        return test_type == "python_function"


class AgenticTestExecutor(TestExecutor):
    """Executor for agentic layer tests (LLM workflow generation)."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agentic test by generating workflow from prompt then validating."""
        from ..utilities.workflow_helpers import generate_only
        
        if not test_case.user_prompt:
            return {
                'success': False,
                'error': 'Agentic test requires user_prompt'
            }
        
        try:
            # Generate workflow from prompt with conversation isolation
            from uuid import uuid4
            test_conversation_id = f"test_agentic_{test_case.id}_{uuid4()}"
            
            tool_catalog = context.get('tool_catalog', {})
            spec = await generate_only(
                test_case.user_prompt, 
                tool_catalog,
                conversation_id=test_conversation_id
            )
            
            if not spec:
                return {
                    'success': False,
                    'error': 'Failed to generate workflow from prompt'
                }
            
            # Now validate the generated workflow against expectations
            expected = test_case.expected_result or {}
            validation_passed = self._validate_generated_workflow(spec, expected, tool_catalog)
            
            return {
                'success': validation_passed,
                'generated_workflow': spec.title if spec else None,
                'nodes_generated': len(spec.nodes) if spec else 0,
                'edges_generated': len(spec.edges) if spec else 0,
                'spec_generated': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Agentic test error: {str(e)}'
            }
    
    def _validate_generated_workflow(self, spec: 'WorkflowSpec', expected: Dict[str, Any], tool_catalog: Dict[str, Any]) -> bool:
        """Validate generated workflow against expected results."""
        if not expected:
            return True  # No expectations = success
        
        # Basic workflow validation
        if "workflow_generated" in expected:
            if not expected["workflow_generated"]:
                return False  # Expected no workflow but got one
        
        # Check for decision nodes
        if "has_decision_nodes" in expected:
            decision_nodes = [n for n in spec.nodes if n.type == "decision" or 
                            (hasattr(n.data, 'tools') and n.data.tools and 'conditional_gate' in n.data.tools)]
            actual = len(decision_nodes) > 0
            expected_val = expected["has_decision_nodes"]
            if actual != expected_val:
                print(f"   ❌ has_decision_nodes: expected={expected_val}, actual={actual}")
                return False
            print(f"   ✅ has_decision_nodes: {actual}")
        
        # Check decision edges have route_index
        if "decision_edges_have_route_index" in expected:
            decision_nodes = [n for n in spec.nodes if n.type == "decision" or 
                            (hasattr(n.data, 'tools') and n.data.tools and 'conditional_gate' in n.data.tools)]
            edges_with_route_index = 0
            total_decision_edges = 0
            
            for decision_node in decision_nodes:
                node_edges = [e for e in spec.edges if e.source == decision_node.id]
                total_decision_edges += len(node_edges)
                
                for edge in node_edges:
                    if edge.data and hasattr(edge.data, 'route_index') and edge.data.route_index is not None:
                        edges_with_route_index += 1
            
            actual = edges_with_route_index > 0
            expected_val = expected["decision_edges_have_route_index"]
            if actual != expected_val:
                print(f"   ❌ decision_edges_have_route_index: expected={expected_val}, actual={actual}")
                return False
            print(f"   ✅ decision_edges_have_route_index: {actual}")
        
        return True
    
    def can_handle(self, test_type: str) -> bool:
        """Handle agentic workflow generation tests."""
        return test_type == "agentic_generation"


class ConversionUtilsExecutor(TestExecutor):
    """Executor for testing conversion utilities using factory pattern."""
    
    def __init__(self):
        self.factory = TestExecutorFactory()
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all conversion test handlers."""
        handlers = [
            TestHandlerSpec(
                name="auto_detection",
                keywords=["auto", "detection"],
                handler_func="_test_auto_detection",
                description="Test automatic data type detection and conversion"
            ),
            TestHandlerSpec(
                name="tool_catalog",
                keywords=["tool", "catalog"],
                handler_func="_test_tool_catalog_conversion", 
                description="Test tool catalog to LLM prompt conversion"
            ),
            TestHandlerSpec(
                name="validation_errors",
                keywords=["validation", "error"],
                handler_func="_test_validation_errors_conversion",
                description="Test validation errors to LLM prompt conversion"
            ),
            TestHandlerSpec(
                name="tool_results",
                keywords=["tool", "result"],
                handler_func="_test_tool_results_conversion",
                description="Test tool usage results to LLM prompt conversion"
            )
        ]
        
        for handler in handlers:
            self.factory.register_handler(handler)
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conversion utilities test using factory pattern."""
        try:
            from ..utilities.conversion_utils import get, ConversionUtils
            
            # Find the right handler for this test
            handler_spec = self.factory.find_handler(test_case.name)
            
            if not handler_spec:
                available_handlers = [h.name for h in self.factory.get_all_handlers()]
                return {
                    'success': False,
                    'error': f'No handler found for test: {test_case.name}. Available handlers: {available_handlers}'
                }
            
            # Get the handler method and execute it
            handler_method = getattr(self, handler_spec.handler_func, None)
            if not handler_method:
                return {
                    'success': False,
                    'error': f'Handler method {handler_spec.handler_func} not found'
                }
            
            # Execute the handler
            result = await handler_method()
            result['handler_used'] = handler_spec.name
            result['handler_description'] = handler_spec.description
            return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Conversion test failed: {str(e)}'
            }
    
    async def _test_auto_detection(self) -> Dict[str, Any]:
        """Test auto-detection functionality."""
        from ..utilities.conversion_utils import get
        
        # Test tool catalog detection
        tool_catalog = {
            "searxng_search": {"description": "Search the web"},
            "calculator": {"description": "Perform calculations"}
        }
        prompt = get(tool_catalog)
        if "searxng_search" not in prompt or "calculator" not in prompt:
            return {'success': False, 'error': 'Tool catalog auto-detection failed'}
        
        # Test validation errors detection
        validation_errors = [["Missing field"], ["Invalid type"]]
        prompt = get(validation_errors)
        if "Validation Errors" not in prompt or "Missing field" not in prompt:
            return {'success': False, 'error': 'Validation errors auto-detection failed'}
        
        return {
            'success': True,
            'message': 'Auto-detection working correctly for all data types'
        }
    
    async def _test_tool_catalog_conversion(self) -> Dict[str, Any]:
        """Test tool catalog conversion."""
        from ..utilities.conversion_utils import ConversionUtils
        
        catalog = {"test_tool": {"description": "Test tool"}}
        prompt = ConversionUtils.tool_catalog_to_llm_prompt(catalog)
        
        if "test_tool" not in prompt or "Test tool" not in prompt:
            return {'success': False, 'error': 'Tool catalog conversion failed'}
        
        return {'success': True, 'message': 'Tool catalog conversion working'}
    
    async def _test_validation_errors_conversion(self) -> Dict[str, Any]:
        """Test validation errors conversion."""
        from ..utilities.conversion_utils import ConversionUtils
        
        errors = [["Error 1"], ["Error 2"]]
        prompt = ConversionUtils.validation_errors_to_llm_prompt(errors)
        
        if "Error 1" not in prompt or "Error 2" not in prompt:
            return {'success': False, 'error': 'Validation errors conversion failed'}
        
        return {'success': True, 'message': 'Validation errors conversion working'}
    
    async def _test_tool_results_conversion(self) -> Dict[str, Any]:
        """Test tool results conversion."""
        from ..utilities.conversion_utils import ConversionUtils
        
        results = [{"tool_name": "test", "result": "success", "tool_args": {"param": "value"}}]
        prompt = ConversionUtils.tool_usage_results_to_llm(results)
        
        if "test" not in prompt or "success" not in prompt:
            return {'success': False, 'error': 'Tool results conversion failed'}
        
        return {'success': True, 'message': 'Tool results conversion working'}
    
    def can_handle(self, test_type: str) -> bool:
        """Handle conversion utilities tests."""
        return test_type == "conversion_utils"


class WorkflowExecutionExecutor(TestExecutor):
    """Executor for full workflow execution tests (orchestration layer)."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full workflow execution test."""
        try:
            from ..utilities.workflow_helpers import plan_and_execute, generate_only
            from ..agent_methods.data_models.workflow_spec import WorkflowSpec
            
            if not test_case.user_prompt:
                return {
                    'success': False,
                    'error': 'No user_prompt provided for workflow execution test'
                }
            
            # Get tool catalog from context
            tool_catalog = context.get('tool_catalog', {})
            
            # Execute the full workflow (generation + execution)
            execution_result = await plan_and_execute(
                prompt=test_case.user_prompt,
                tool_catalog=tool_catalog,
                max_retries=3,
                debug=True
            )
            
            if not execution_result:
                return {
                    'success': False,
                    'error': 'Failed to execute workflow from prompt'
                }
            
            # Extract workflow spec and execution state
            workflow_spec = execution_result.get('workflow_spec')
            final_state = execution_result.get('execution_result')
            
            if not workflow_spec or not final_state:
                return {
                    'success': False,
                    'error': 'Invalid execution result structure'
                }
            
            # Analyze results against expected outcome
            expected = test_case.expected_result or {}
            execution_stats = execution_result.get('execution_stats', {})
            
            executed_count = execution_stats.get('executed_nodes', 0)
            skipped_count = execution_stats.get('total_nodes', 0) - executed_count
            total_count = execution_stats.get('total_nodes', 0)
            
            actual_result = {
                'workflow_generated': True,
                'execution_completed': execution_result.get('success', False),
                'executed_nodes': executed_count,
                'skipped_nodes': skipped_count,
                'total_nodes': total_count,
                'execution_pattern': self._analyze_execution_pattern(executed_count, skipped_count, workflow_spec),
                'route_behavior': self._analyze_routing_behavior(workflow_spec)
            }
            
            # Check expectations
            success = True
            validation_details = []
            
            if 'should_execute_downstream' in expected:
                downstream_executed = actual_result['executed_nodes'] > 2  # More than just input + gate
                if expected['should_execute_downstream'] != downstream_executed:
                    success = False
                    validation_details.append(f"Expected downstream execution: {expected['should_execute_downstream']}, got: {downstream_executed}")
            
            if 'expected_node_execution_count' in expected:
                if expected['expected_node_execution_count'] != actual_result['executed_nodes']:
                    success = False
                    validation_details.append(f"Expected {expected['expected_node_execution_count']} executed nodes, got {actual_result['executed_nodes']}")
            
            if 'expected_skipped_nodes' in expected:
                if expected['expected_skipped_nodes'] != actual_result['skipped_nodes']:
                    success = False
                    validation_details.append(f"Expected {expected['expected_skipped_nodes']} skipped nodes, got {actual_result['skipped_nodes']}")
            
            # Record test result in workflow metadata if workflow was generated
            if workflow_spec and hasattr(workflow_spec, 'add_test_result'):
                workflow_spec.add_test_result(
                    test_id=test_case.id,
                    test_name=test_case.name,
                    passed=success and test_case.should_pass,
                    execution_details=actual_result,
                    error_message=None if success else "; ".join(validation_details)
                )
            
            return {
                'success': success,
                'workflow_spec': workflow_spec.model_dump() if workflow_spec else None,
                'execution_result': execution_result,
                'analysis': actual_result,
                'validation_details': validation_details,
                'test_passed': success and test_case.should_pass,
                'test_alignment_updated': workflow_spec is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Workflow execution test failed: {str(e)}',
                'test_passed': False
            }
    
    def _analyze_execution_pattern(self, executed_count: int, skipped_count: int, workflow_spec: 'WorkflowSpec') -> str:
        """Analyze the execution pattern."""
        if skipped_count > 0:
            return 'conditional_gate_trigger' if executed_count > 2 else 'conditional_gate_no_trigger'
        else:
            return 'full_execution'
    
    def _analyze_routing_behavior(self, workflow_spec: 'WorkflowSpec') -> str:
        """Analyze routing behavior."""
        # Look for conditional edges in workflow
        has_conditional_edges = any(
            edge.data and hasattr(edge.data, 'route_index') and edge.data.route_index is not None
            for edge in workflow_spec.edges
        )
        
        if has_conditional_edges:
            conditional_edges = [e for e in workflow_spec.edges if e.data and hasattr(e.data, 'route_index') and e.data.route_index is not None]
            return 'single_edge_conditional' if len(conditional_edges) == 1 else 'multi_branch_conditional'
        else:
            return 'no_conditional_routing'
    
    def can_handle(self, test_type: str) -> bool:
        """Check if this executor can handle workflow execution tests."""
        return test_type == "workflow_execution"


class TestExecutorRegistry:
    """Registry for test executors."""
    
    def __init__(self):
        self.executors = []
        
        # Register default executors
        self.register(WorkflowValidationExecutor())
        self.register(PythonFunctionExecutor())
        self.register(AgenticTestExecutor())
        self.register(ConversionUtilsExecutor())
        self.register(WorkflowExecutionExecutor())
    
    def register(self, executor: TestExecutor):
        """Register a test executor."""
        self.executors.append(executor)
    
    def get_executor(self, test_type: str) -> Optional[TestExecutor]:
        """Get executor for a test type."""
        for executor in self.executors:
            if executor.can_handle(test_type):
                return executor
        return None
    
    def list_supported_types(self) -> Dict[str, str]:
        """List all supported test types."""
        types = {}
        test_types_to_check = [
            'workflow_validation', 
            'python_function', 
            'agentic_generation',
            'conversion_utils',
            'workflow_execution',
            'database_test',
            'api_test'
        ]
        
        for executor in self.executors:
            executor_name = executor.__class__.__name__
            if hasattr(executor, 'can_handle'):
                for test_type in test_types_to_check:
                    if executor.can_handle(test_type):
                        types[test_type] = executor_name
        return types


# Global registry instance
_registry = TestExecutorRegistry()

def get_test_executor(test_type: str) -> Optional[TestExecutor]:
    """Get test executor for a given test type."""
    return _registry.get_executor(test_type)

def register_test_executor(executor: TestExecutor):
    """Register a custom test executor."""
    _registry.register(executor)

def list_test_types() -> Dict[str, str]:
    """List all supported test types."""
    return _registry.list_supported_types()


# Example custom executor for demonstration
class DatabaseTestExecutor(TestExecutor):
    """Example executor for database tests."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database test (placeholder)."""
        return {
            'success': True,
            'message': 'Database test executor placeholder'
        }
    
    def can_handle(self, test_type: str) -> bool:
        return test_type == "database_test"


class APITestExecutor(TestExecutor):
    """Example executor for API tests."""
    
    async def execute_test(self, test_case: 'WorkflowTestCase', context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API test (placeholder)."""
        return {
            'success': True,
            'message': 'API test executor placeholder'
        }
    
    def can_handle(self, test_type: str) -> bool:
        return test_type == "api_test"


# Auto-register executors
register_test_executor(WorkflowValidationExecutor())
register_test_executor(PythonFunctionExecutor())
register_test_executor(AgenticTestExecutor())
register_test_executor(WorkflowExecutionExecutor())
register_test_executor(DatabaseTestExecutor())
register_test_executor(APITestExecutor())