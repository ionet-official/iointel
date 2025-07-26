# Test Suite Migration Guide

## Overview

This guide shows how to migrate existing tests to use the new centralized workflow test repository system.

## Migration Phases

### Phase 1: Update Existing Tests (Minimal Changes)
Just change imports and fixture names - tests keep working immediately.

### Phase 2: Convert to Layered Architecture  
Move tests into proper layer classifications.

### Phase 3: Use Smart Fixtures
Replace custom data with smart fixtures that understand context.

## Phase 1: Minimal Migration

### Before (old test):
```python
# tests/test_workflow_planner.py
@pytest.fixture
def tool_catalog():
    return {
        "weather_api": {...},
        "send_email": {...}
    }

def test_something(tool_catalog):
    # test logic
```

### After (minimal change):
```python
# tests/test_workflow_planner.py  
# Remove local fixture, use centralized one
def test_something(mock_tool_catalog):  # <-- Just change fixture name
    # test logic remains the same
```

### Before (old real tools):
```python
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env

@pytest.fixture
def real_tools():
    load_tools_from_env('creds.env')
    return create_tool_catalog()
```

### After (minimal change):
```python
# Just use the centralized fixture
def test_something(real_tool_catalog):  # <-- Already available!
    # test logic remains the same
```

## Phase 2: Layer Classification

### Before (mixed concerns):
```python  
@pytest.mark.asyncio
async def test_workflow_generation_and_validation():
    # This test does EVERYTHING - generation + validation + execution
    planner = WorkflowPlanner()
    workflow = await planner.generate_workflow(query="stock agent")
    
    # Validation logic
    issues = workflow.validate_structure(tool_catalog)
    assert len(issues) == 0
    
    # Execution logic  
    result = dag_executor.execute(workflow)
    assert result.success
```

### After (layered):
```python
# Split into focused layer tests

class TestAgenticLayer:
    @pytest.mark.asyncio
    async def test_stock_workflow_generation(self, stock_analysis_prompts, real_tool_catalog):
        # ONLY test LLM generation
        planner = WorkflowPlanner()
        for prompt in stock_analysis_prompts:
            workflow = await planner.generate_workflow(query=prompt, tool_catalog=real_tool_catalog)
            assert workflow is not None

class TestLogicalLayer:  
    def test_workflow_validation(self, validation_test_cases, mock_tool_catalog):
        # ONLY test validation logic
        for test_case in validation_test_cases:
            workflow = WorkflowSpec(**test_case.workflow_spec)
            issues = workflow.validate_structure(mock_tool_catalog)
            assert (len(issues) == 0) == test_case.should_pass

class TestOrchestrationLayer:
    def test_workflow_execution(self, pipeline_execution_cases):  
        # ONLY test execution
        for test_case in pipeline_execution_cases:
            # Execute workflow and verify results
            pass
```

## Phase 3: Smart Fixtures

### Before (hardcoded data):
```python
def test_conditional_routing():
    workflow_data = {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "buy", "type": "agent", "label": "Buy Agent"},
            {"id": "sell", "type": "agent", "label": "Sell Agent"}
        ],
        "edges": [
            {"source": "decision", "target": "buy", "data": {"route_index": 0}},
            {"source": "decision", "target": "sell", "data": {"route_index": 1}}
        ]
    }
    # Test logic using hardcoded data
```

### After (smart fixtures):
```python
@pytest.mark.layer("logical")
@pytest.mark.category("conditional_routing")  
def test_conditional_routing(smart_test_data):
    # Gets conditional routing test cases automatically!
    routing_cases = smart_test_data.get('routing_cases', [])
    for test_case in routing_cases:
        workflow_data = test_case.workflow_spec
        # Test logic using centralized test data
```

## Common Migration Patterns

### Pattern 1: Tool Catalog Migration
```python
# OLD - every test has its own
@pytest.fixture
def tools():
    return {"weather": {...}, "email": {...}}

# NEW - use centralized
def test_something(mock_tool_catalog):  # or real_tool_catalog
    pass
```

### Pattern 2: Workflow Spec Migration  
```python
# OLD - hardcoded in each test
def test_workflow():
    spec = WorkflowSpec(nodes=[...], edges=[...])

# NEW - from test repository
def test_workflow(conditional_routing_cases):
    for test_case in conditional_routing_cases:
        spec = WorkflowSpec(**test_case.workflow_spec)
```

### Pattern 3: User Prompt Migration
```python  
# OLD - scattered prompts
def test_stock_generation():
    query = "stock agent"
    
# NEW - centralized prompts
def test_stock_generation(stock_analysis_prompts):
    for prompt in stock_analysis_prompts:
        # Test with all known stock prompts
```

## File-by-File Migration Examples

### tests/test_workflow_planner.py
- ✅ Keep existing fixture names for backward compatibility
- ✅ Add new tests using smart fixtures  
- ✅ Gradually convert existing tests to use centralized data

### tests/routing/test_conditional_routing.py  
- ✅ Move to TestLogicalLayer in centralized system
- ✅ Use conditional_routing_cases fixture
- ✅ Remove duplicate test data

### tests/workflows/test_*.py
- ✅ Classify by layer (logical/agentic/orchestration/feedback)  
- ✅ Use appropriate smart fixtures
- ✅ Remove scattered data definitions