# TODO: Unify Validation Systems

## Current State
We have two separate validation systems:

1. **Tool Catalog Generation** (`run_workflow_planner.py`)
   - Uses pydantic-ai's `function_schema()` 
   - Correctly handles `**kwargs`
   - Provides rich descriptions for LLM
   - ✅ Working well

2. **Runtime Validation** (`func_metadata.py`)
   - Homegrown pydantic model generation
   - Incorrectly treats `**kwargs` as required field
   - Used during actual function execution
   - ❌ Breaks tools with `**kwargs` like user_input

## The Problem
- Maintaining two validation systems that can diverge
- The old system doesn't handle `**kwargs` properly
- Validation errors occur at runtime even when LLM generates correct params

## Quick Fix Applied
Modified `func_metadata.py` to skip `**kwargs` parameters:
```python
# Skip **kwargs parameters - they don't need validation
if param.kind == inspect.Parameter.VAR_KEYWORD:
    continue
```

## Proper Solution (TODO)
Replace `func_metadata.py` with pydantic-ai's validation:
```python
# Instead of:
metadata = func_metadata(tool_function)
validated = metadata.arg_model.model_validate(args)

# Use:
func_schema = function_schema(tool_function, ...)
validated = func_schema.validator.validate_python(args)
```

## Benefits of Unification
- Single source of truth for tool validation
- Consistent parameter handling everywhere
- Less code to maintain
- Leverage pydantic-ai's ongoing improvements

## Implementation Notes
- Tools are already registered with pydantic-ai compatibility
- Need to update `ToolWrapper.run()` to use pydantic-ai validation
- Ensure backward compatibility during transition