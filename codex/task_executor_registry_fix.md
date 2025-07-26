# Task Executor Registry Architecture Fix
*Date: 2025-07-26*
*Status: Implemented*

## Overview

This document describes the architectural fix for the Task Executor Registry system, addressing the core issue where fundamental task executors were not properly registered, causing "No executor registered for task type" errors during workflow execution.

## Problem Statement

### Original Issue
- **Error**: `No executor registered for task type: data_source`
- **Error**: `No executor registered for task type: decision`
- **Root Cause**: Empty `TASK_EXECUTOR_REGISTRY` at runtime
- **Impact**: Complete workflow execution failure

### Architectural Confusion
The system had two layers of executors:
1. **Core Business Logic Executors** (`chainables.py`) - The actual task execution logic
2. **Web Layer Wrappers** (`workflow_server.py`) - UI-specific executor wrappers

The core executors were not properly registered with the `@register_custom_task` decorator, leading to an empty registry.

## Solution Architecture

### 1. Core Executor Registration (`chainables.py`)

Added proper decorator registration for all fundamental task types:

```python
@register_custom_task("data_source")  # Core data_source executor  
@register_custom_task("tool")  # Backward compatibility
async def execute_tool_task(task_metadata, objective, agents, execution_metadata):
    """Portable, backend-agnostic tool executor for 'tool' nodes."""
    # Implementation handles both data_source and legacy tool types

@register_custom_task("agent")  # Core agent executor
async def execute_agent_task(task_metadata, objective, agents, execution_metadata):
    """Generic agent task executor that handles type='agent' tasks from WorkflowSpec."""
    # Converts AgentParams to Agent instances
    # Handles agent instructions and tool execution

@register_custom_task("decision")  # Core decision executor
async def execute_decision_task(task_metadata, objective, agents, execution_metadata):
    """Core decision executor - delegates to agent executor for decision agents."""
    # Decision nodes are agents with routing tools
    return await execute_agent_task(task_metadata, objective, agents, execution_metadata)
```

### 2. Registry System (`registries.py` + `decorators.py`)

The registration system provides a clean separation of concerns:

```python
# Global registry mapping task types to executor functions
TASK_EXECUTOR_REGISTRY: Dict[str, Callable] = {}

def register_custom_task(task_type: str, chainable: bool = True):
    """Decorator that registers a custom task executor for a given task type."""
    def decorator(tool_fn: Callable):
        # Register the executor function for later task execution
        TASK_EXECUTOR_REGISTRY[task_type] = tool_fn
        return tool_fn
    return decorator
```

### 3. Task Type Resolution (`workflow.py`)

The workflow execution system resolves task types and dispatches to appropriate executors:

```python
async def run_task(self, task, default_text, default_agents, conversation_id, **kwargs):
    # Determine task type from task dictionary
    task_type = task.get("type") or task.get("name")
    
    # Look up executor in registry
    executor = TASK_EXECUTOR_REGISTRY.get(task_type)
    if executor is None:
        raise ValueError(f"No executor registered for task type: {task_type}")
    
    # Execute with proper parameters
    result = executor(
        task_metadata=task_metadata,
        objective=text_for_task,
        agents=agents_for_task,
        execution_metadata=execution_metadata,
    )
```

## Executor Implementations

### Data Source Executor
```python
@register_custom_task("data_source")
@register_custom_task("tool")  # Backward compatibility
async def execute_tool_task(task_metadata, objective, agents, execution_metadata):
    """Handles data_source nodes (formerly 'tool' nodes)."""
    tool_name = task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    
    # Fetch tool from registry and execute
    tool = TOOLS_REGISTRY.get(tool_name)
    result = tool.run(config_with_metadata)
    return result
```

### Agent Executor
```python
@register_custom_task("agent")
async def execute_agent_task(task_metadata, objective, agents, execution_metadata):
    """Handles agent nodes with proper AgentParams conversion."""
    agent_instructions = task_metadata.get("agent_instructions", "")
    
    # Convert AgentParams to Agent instances if needed
    if isinstance(agents[0], AgentParams):
        agents_to_use = [create_agent(ap) for ap in agents]
    
    # Execute agent with context and instructions
    response = await run_agents(
        objective=task_objective,
        agents=agents_to_use,
        context=context,
        conversation_id=conversation_id,
        result_format=result_format,
    ).execute()
    
    return response
```

### Decision Executor
```python
@register_custom_task("decision")
async def execute_decision_task(task_metadata, objective, agents, execution_metadata):
    """Decision nodes are agents with routing tools."""
    # Delegate to agent executor since decision nodes are just specialized agents
    return await execute_agent_task(task_metadata, objective, agents, execution_metadata)
```

## Decision Node Agent Configuration Fix

### Problem
Decision nodes with `type="decision"` were not getting proper agent instances created:
- **Error**: "No agent configuration provided for agent node"
- **Cause**: DAG executor only auto-created agents for `type="agent"`, not `type="decision"`

### Solution
Updated DAG executor to treat decision nodes as agents:

```python
# In dag_executor.py
if (node.type in ["agent", "decision"]) and len(node_agents) == 0:
    # Auto-create agents from WorkflowSpec node data
    # Decision nodes are also agents (with routing tools)
    hydrated_agents = self._hydrate_agents_from_node(node)

# Also updated _hydrate_agents_from_node method
if node.type not in ["agent", "decision"] or not node.data.agent_instructions:
    return None
```

## Registry Population Flow

### 1. Module Import Time
When `chainables.py` is imported, decorators execute and populate the registry:

```python
# At import time, these decorators run:
@register_custom_task("data_source")  # Adds to TASK_EXECUTOR_REGISTRY["data_source"]
@register_custom_task("tool")         # Adds to TASK_EXECUTOR_REGISTRY["tool"] 
@register_custom_task("agent")        # Adds to TASK_EXECUTOR_REGISTRY["agent"]
@register_custom_task("decision")     # Adds to TASK_EXECUTOR_REGISTRY["decision"]
```

### 2. Runtime Execution
During workflow execution, the registry is consulted:

```python
# In workflow.py run_task method
task_type = task.get("type")  # e.g., "decision"
executor = TASK_EXECUTOR_REGISTRY.get(task_type)  # Gets execute_decision_task
result = executor(task_metadata, objective, agents, execution_metadata)
```

## Separation of Concerns

### Core Layer (`chainables.py`)
- **Purpose**: Business logic execution
- **Scope**: Task type → execution function mapping
- **Dependencies**: Core models, tool registry, agent factory
- **Registration**: `@register_custom_task` decorators

### Web Layer (`workflow_server.py`) 
- **Purpose**: UI-specific features and wrappers
- **Scope**: HTTP endpoints, UI state management
- **Dependencies**: Core layer + web framework
- **Registration**: Separate web-specific registry

### Benefits of This Architecture
1. **Core Independence**: Business logic doesn't depend on web layer
2. **Testability**: Core executors can be tested in isolation  
3. **Reusability**: Core executors work in any context (CLI, API, tests)
4. **Maintainability**: Clear separation reduces coupling

## Backward Compatibility

### Tool vs Data Source
The system maintains compatibility with both naming conventions:
```python
@register_custom_task("data_source")  # New preferred name
@register_custom_task("tool")         # Legacy compatibility
async def execute_tool_task(...):     # Handles both types
```

### Agent Types
Both regular agents and decision agents use the same underlying logic:
```python
# Agent nodes: direct execution
await execute_agent_task(...)

# Decision nodes: delegated execution (same logic)
await execute_decision_task(...)  # Calls execute_agent_task internally
```

## Testing Integration

### Executor Validation
The unified test system can now properly test all task types:
```python
# All these task types now have registered executors
test_cases = [
    {"type": "data_source", "tool_name": "yfinance_get_stock_price"},
    {"type": "agent", "agent_instructions": "Analyze the data"},
    {"type": "decision", "agent_instructions": "Route based on condition"}
]
```

### Error Prevention
The fix prevents the most common workflow execution failure mode:
- Before: `No executor registered for task type: decision` → 100% failure
- After: Proper executor dispatch → Normal execution flow

## Performance Impact

### Registration Overhead
- **Cost**: Minimal - decorators run once at import time
- **Benefit**: Zero runtime lookup overhead
- **Memory**: Small dictionary with function references

### Execution Overhead  
- **Decision Delegation**: Minimal function call overhead
- **Agent Creation**: Amortized over agent lifecycle
- **Tool Execution**: No change from previous implementation

## Future Considerations

### Dynamic Registration
Could support runtime executor registration for plugins:
```python
# Runtime registration for custom task types
register_executor_dynamically("custom_ml_task", custom_ml_executor)
```

### Executor Validation
Could add validation that all referenced task types have executors:
```python
def validate_workflow_executors(workflow_spec: WorkflowSpec):
    missing_executors = []
    for node in workflow_spec.nodes:
        if node.type not in TASK_EXECUTOR_REGISTRY:
            missing_executors.append(node.type)
    return missing_executors
```

### Registry Introspection
Could add debugging tools to inspect registry state:
```python
def debug_executor_registry():
    return {
        "registered_types": list(TASK_EXECUTOR_REGISTRY.keys()),
        "executor_functions": [f.__name__ for f in TASK_EXECUTOR_REGISTRY.values()]
    }
```

## Related Files

- `/iointel/src/chainables.py` - Core executor implementations and registration
- `/iointel/src/utilities/registries.py` - Registry definitions
- `/iointel/src/utilities/decorators.py` - Registration decorator
- `/iointel/src/workflow.py` - Task type resolution and dispatch
- `/iointel/src/utilities/dag_executor.py` - Decision node agent creation fix
- `/iointel/src/web/workflow_server.py` - Web layer executor wrappers

## Verification

### Registry Population Check
```python
from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
print("Registered executors:", list(TASK_EXECUTOR_REGISTRY.keys()))
# Output: ['data_source', 'tool', 'agent', 'decision', ...]
```

### Workflow Execution Test
```bash
# Gate pattern tests now pass executor lookup
python run_unified_tests.py --tags gate_pattern
# No more "No executor registered" errors
```