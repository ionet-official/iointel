# Workflow Ontology Mismatch Analysis

## Problem Summary

The iointel codebase has **three different workflow ontologies** operating simultaneously, creating inconsistencies in task identification and execution.

## The Three Ontologies

### 1. WorkflowSpec (Modern DAG-based)
**Location**: `/src/agent_methods/data_models/workflow_spec.py`
**Purpose**: LLM-generated workflows, React Flow rendering
**Schema**: 
```python
class NodeSpec(BaseModel):
    id: str              # REQUIRED - unique identifier
    type: Literal["tool", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None
    runtime: Dict = Field(default_factory=dict)
```

### 2. WorkflowDefinition (Legacy Task-based)
**Location**: `/src/agent_methods/data_models/datamodels.py`
**Purpose**: YAML serialization, traditional workflows
**Schema**:
```python
class TaskDefinition(BaseModel):
    task_id: str         # REQUIRED - unique identifier
    name: str
    type: str = "custom"
    objective: Optional[str] = None
    agents: Optional[Union[List[AgentParams], AgentSwarm]] = None
    task_metadata: Optional[Dict[str, Any]] = None
    execution_metadata: Optional[Dict[str, Any]] = None
```

### 3. Workflow.add_task() (Runtime Dict-based)
**Location**: `/src/workflow.py`
**Purpose**: Programmatic workflow construction
**Schema**: Plain Python dicts with **NO Pydantic validation**
```python
# Current usage - no validation!
wf.add_task({
    'type': 'agent',           # Optional
    'name': 'riddle_task',     # Optional - not checked by _get_task_key()
    'objective': '...',        # Optional
    'task_metadata': {...}     # Optional
})
```

## The Mismatch

### Task Key Resolution Bug
The `_get_task_key()` function was designed for validated schemas but is used with unvalidated dicts:

```python
def _get_task_key(task: dict) -> str:
    return (
        task.get("task_id")                    # From TaskDefinition/NodeSpec
        or task.get("task_metadata", {}).get("name")  # Nested field
        or task.get("type")                    # Fallback - gives 'agent'
        or "task"
    )
    # Missing: task.get("name")  ← Top-level name field!
```

**Result**: Tasks with `name: 'riddle_task'` get stored as `'agent'` in workflow results.

### AgentResultFormat Parameter Flow
The new AgentResultFormat system works correctly but revealed this ontology mismatch during testing.

## Impact Analysis

1. **Task Identification**: Tasks are incorrectly keyed by type instead of name
2. **Workflow Results**: Results stored under wrong keys (`'agent'` vs `'riddle_task'`)
3. **Developer Confusion**: Three different ways to define the same concept
4. **Validation Gap**: Runtime dicts have no schema enforcement
5. **Maintenance Burden**: Changes must be made in multiple places

## Solutions Implemented

### 1. Quick Fix: Update _get_task_key()
```python
def _get_task_key(task: dict) -> str:
    return (
        task.get("task_id")
        or task.get("name")              # ← Added top-level name
        or task.get("task_metadata", {}).get("name")
        or task.get("type")
        or "task"
    )
```

### 2. Medium Fix: Standardize on One Ontology
**Recommendation**: Use `TaskDefinition` as the canonical format
- Add conversion methods: `dict → TaskDefinition → WorkflowSpec`
- Deprecate direct dict usage in `Workflow.add_task()`
- Maintain backward compatibility during transition

### 3. Long-term Fix: Add Pydantic Validation
**Goal**: Enforce consistent schemas at runtime
```python
def add_task(self, task: Union[dict, TaskDefinition]):
    if isinstance(task, dict):
        # Validate and convert to TaskDefinition
        task = TaskDefinition.model_validate(task)
    self.tasks.append(task)
```

## Migration Strategy

1. **Phase 1**: Fix `_get_task_key()` (immediate)
2. **Phase 2**: Add optional Pydantic validation with warnings
3. **Phase 3**: Make validation required, deprecate dict format
4. **Phase 4**: Remove dict support, standardize on TaskDefinition

## Files Modified

- `/src/workflow.py` - Fixed _get_task_key()
- `/src/agent_methods/data_models/datamodels.py` - Added AgentResultFormat
- `/src/agents.py` - Updated result formatting
- `/src/chainables.py` - Updated to use AgentResultFormat
- `/src/utilities/runners.py` - Fixed parameter propagation

## Testing

The AgentResultFormat system now works correctly:
- `chat`: `['result']`
- `chat_w_tools`: `['result', 'tool_usage_results']`
- `workflow`: `['result', 'conversation_id', 'tool_usage_results']`
- `full`: `['result', 'conversation_id', 'tool_usage_results', 'full_result']`

## Recommendations

1. **Immediate**: Use the quick fix for production stability
2. **Short-term**: Begin migration to TaskDefinition-based workflows
3. **Long-term**: Implement unified workflow ontology with proper validation
4. **Documentation**: Update workflow creation examples to use consistent patterns

## Related Issues

- AgentResultFormat parameter propagation through workflow execution
- Tool parameter validation for optional vs required parameters
- execution_metadata handling across different workflow types