"""
Test Executor Architecture
==========================

Pluggable test executors for different test types in the unified test system.
This allows extending the test framework to support ANY kind of test - not just workflows.

Architecture:
- Base TestExecutor interface
- Concrete executors for different test types
- Registry system for plugging in new executors
- Backward compatibility with existing workflow tests
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import importlib
import inspect
from pathlib import Path


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
        if not expected:
            return True  # No expectations means any result is ok
            
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
                
        return True
    
    def can_handle(self, test_type: str) -> bool:
        """Handle Python function tests."""
        return test_type == "python_function"


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
        for executor in self.executors:
            executor_name = executor.__class__.__name__
            # This is a simplified check - in reality you'd want better introspection
            if hasattr(executor, 'can_handle'):
                if executor.can_handle('workflow_validation'):
                    types['workflow_validation'] = executor_name
                if executor.can_handle('python_function'):
                    types['python_function'] = executor_name
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
register_test_executor(WorkflowExecutionExecutor())
register_test_executor(DatabaseTestExecutor())
register_test_executor(APITestExecutor())