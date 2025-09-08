# Test File Reorganization

The following test files were moved from the root directory to appropriate test folders:

## Execution Tests (tests/execution/)
- `test_dag_debug.py` - DAG debugging utilities
- `test_dag_fix.py` - DAG execution fixes

## Routing Tests (tests/routing/)
- `test_decision_routing_new.py` - Decision node routing logic
- `test_routing_gate.py` - Routing gate functionality

## Integration Tests (tests/integration/)
- `test_decision_prompt_injection.py` - SLA enforcement with prompt injection
- `test_full_system_fixes.py` - Full system integration tests
- `test_unified_fixes.py` - Unified validation, retry, and SLA tests

## Workflow Pattern Tests (tests/workflows/patterns/)
- `test_user_input_fix.py` - User input flow tests
- `test_user_input_workflow.py` - User input with plan_and_execute

## Workflow Tests (tests/workflows/)
- `test_workflow_helpers.py` - Workflow helper function tests

## Core Tests (tests/)
- `test_new_workflow_spec.py` - WorkflowSpec refactor tests
- `test_new_repository.py` - Test repository functionality
- `test_agent_creation_from_spec.py` - Agent creation from specifications
- `test_conversion_utils.py` - (already in correct location)

All tests are now properly organized in the test directory structure.
