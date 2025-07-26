# Tool Registry Refactor Plan

## Phase 1: Enhanced Centralized Tool Resolution

### 1.1 Update `tool_registry_utils.py`

```python
# iointel/src/utilities/tool_registry_utils.py

from typing import Union, Tuple, Dict, Any, Optional
from ..agent_methods.data_models.datamodels import Tool
from .registries import TOOLS_REGISTRY
from ..utilities.helpers import make_logger

logger = make_logger(__name__)

def resolve_tool(tool_ref: Union[str, Tuple[str, dict], dict]) -> Tool:
    """
    Single point of tool resolution. Always returns a tool from the registry.
    
    Args:
        tool_ref: Can be:
            - str: Tool name (e.g., "run_shell_command")
            - tuple: (tool_name, state_args) for stateful tools
            - dict: Must have 'name' key, all other keys ignored
    
    Returns:
        Tool from registry with fn properly set
    
    Raises:
        ValueError: If tool not found in registry
    """
    # Extract tool name and state args
    if isinstance(tool_ref, str):
        tool_name = tool_ref
        state_args = None
    elif isinstance(tool_ref, tuple) and len(tool_ref) == 2:
        tool_name, state_args = tool_ref
        if not isinstance(tool_name, str):
            raise ValueError(f"Tool name in tuple must be string, got {type(tool_name)}")
    elif isinstance(tool_ref, dict):
        tool_name = tool_ref.get('name')
        state_args = tool_ref.get('state_args')
        if not tool_name:
            raise ValueError("Dict tool reference must have 'name' key")
    else:
        raise ValueError(
            f"Invalid tool reference type: {type(tool_ref)}. "
            f"Expected str, tuple, or dict"
        )
    
    # Look up in registry
    if tool_name not in TOOLS_REGISTRY:
        available = ", ".join(sorted(TOOLS_REGISTRY.keys())[:10])
        raise ValueError(
            f"Tool '{tool_name}' not found in registry. "
            f"Available tools include: {available}..."
        )
    
    tool = TOOLS_REGISTRY[tool_name]
    
    # Validate tool has fn
    if not tool.fn:
        raise ValueError(
            f"Tool '{tool_name}' in registry has no function. "
            f"This indicates a registration error."
        )
    
    # Handle stateful tools if needed
    if state_args and tool.fn_metadata and tool.fn_metadata.stateful:
        # TODO: Implement stateful tool instantiation
        # For now, just return the base tool
        logger.warning(
            f"Stateful tool '{tool_name}' requested with args {state_args}, "
            f"but stateful instantiation not yet implemented"
        )
    
    return tool

def resolve_tools(tool_refs: list) -> list[Tool]:
    """Resolve a list of tool references."""
    return [resolve_tool(ref) for ref in tool_refs]

# DELETE these functions as they add complexity:
# - get_tool_info() 
# - validate_tool_registry()
# - debug_tool_resolution()
```

## Phase 2: Simplify Tool Factory

### 2.1 Refactor `tool_factory.py`

```python
# iointel/src/agent_methods/agents/tool_factory.py

from typing import List
from pydantic import BaseModel
from ..data_models.datamodels import AgentParams, Tool
from ...utilities.tool_registry_utils import resolve_tool
from ...utilities.helpers import make_logger

logger = make_logger(__name__)

# DELETE the rehydrate_tool function entirely

def resolve_tools(
    params: AgentParams,
    tool_instantiator = None  # Keep for backward compat, but ignore
) -> List[Tool]:
    """
    Resolve tools from AgentParams using the centralized registry.
    
    Each tool in params.tools should be:
      - a string: tool name
      - a tuple: (tool_name, state_args) 
      - a dict: must have 'name' key
    
    Everything else is rejected.
    """
    resolved_tools = []
    
    for tool_ref in params.tools:
        try:
            tool = resolve_tool(tool_ref)
            resolved_tools.append(tool)
            logger.debug(f"Resolved tool: {tool.name}")
        except ValueError as e:
            logger.error(f"Failed to resolve tool: {e}")
            raise
    
    return resolved_tools

# DELETE instantiate_stateful_tool function (move to tool_registry_utils if needed)
```

## Phase 3: Simplify Agent Tool Resolution

### 3.1 Update `agents.py`

```python
# iointel/src/agents.py

# In the Agent class:

@classmethod
def _get_registered_tool(
    cls, tool: str | Tool | Callable, allow_unregistered_tools: bool
) -> Tool:
    """DEPRECATED: Use tool_registry_utils.resolve_tool instead."""
    from .utilities.tool_registry_utils import resolve_tool
    
    # For backward compatibility, convert Tool/Callable to registry lookup
    if isinstance(tool, Tool):
        tool = tool.name
    elif callable(tool) and hasattr(tool, '__name__'):
        tool = tool.__name__
    
    try:
        return resolve_tool(tool)
    except ValueError:
        if allow_unregistered_tools and callable(tool):
            return Tool.from_function(tool)
        raise
```

## Phase 4: Fix Workflow YAML Handling

### 4.1 Update `workflow.py`

```python
# iointel/src/workflow.py

# In from_yaml method:

@classmethod
def from_yaml(
    cls,
    yaml_str: str = None,
    file_path: str = None,
    instantiate_agent = None,
    instantiate_tool = None,  # IGNORE this parameter
) -> "Workflow":
    """Load workflow from YAML. Tools are always resolved from registry."""
    if not yaml_str and not file_path:
        raise ValueError("Either yaml_str or file_path must be provided.")
    
    if yaml_str:
        data = yaml.safe_load(yaml_str)
    else:
        data = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
    
    wf_def = WorkflowDefinition(**data)
    
    # Process agents - tools are just names in YAML
    for agent_data in wf_def.agents or []:
        # Agent tools should already be strings or simple refs
        # The create_agent function will resolve them
        pass
    
    # Continue with existing logic...
```

### 4.2 Update `agents_factory.py`

```python
# iointel/src/agent_methods/agents/agents_factory.py

def create_agent(
    params: AgentParams,
    instantiate_agent: Callable[[AgentParams], Agent] | None = None,
    instantiate_tool = None,  # IGNORE - only here for backward compat
) -> Agent:
    """Create an Agent instance. Tools are always resolved from registry."""
    
    # Use simplified tool resolution
    from .tool_factory import resolve_tools
    tools = resolve_tools(params)
    
    # Rest of the function stays the same...
    output_type = params.output_type
    if isinstance(output_type, str):
        output_type = globals().get(output_type) or __builtins__.get(
            output_type, output_type
        )
    
    return (
        instantiate_agent_default if instantiate_agent is None else instantiate_agent
    )(params.model_copy(update={"tools": tools, "output_type": output_type}))
```

## Phase 5: Simplify Tool Serialization

### 5.1 Update `Tool` model

```python
# iointel/src/agent_methods/data_models/datamodels.py

class Tool(BaseModel):
    # ... existing fields ...
    
    def to_yaml_ref(self) -> Union[str, Tuple[str, dict]]:
        """Convert tool to YAML reference (just the name)."""
        if self.fn_metadata and self.fn_metadata.stateful and self.fn_self:
            # For stateful tools, return tuple
            return (self.name, {"state": "serialized_state"})
        return self.name
    
    @classmethod
    def from_yaml_ref(cls, ref: Union[str, Tuple[str, dict]]) -> "Tool":
        """Create tool from YAML reference using registry."""
        from ...utilities.tool_registry_utils import resolve_tool
        return resolve_tool(ref)
```

### 5.2 Update `agent_or_swarm` serialization

```python
# iointel/src/agent_methods/agents/agents_factory.py

def agent_or_swarm(
    agent_obj: Agent | Sequence[Agent], store_creds: bool
) -> list[AgentParams] | AgentSwarm:
    """Serialize agents. Tools are stored as names only."""
    
    def make_params(agent: Agent) -> AgentParams:
        return AgentParams(
            name=agent.name,
            instructions=agent.instructions,
            persona=agent.persona,
            # Tools are just names or (name, args) tuples
            tools=[t.to_yaml_ref() for t in agent.tools],
            model=getattr(agent.model, "model_name", None),
            model_settings=agent.model_settings,
            api_key=get_api_key(agent),
            base_url=agent.base_url,
            memory=agent.memory,
            context=agent.context,
            output_type=agent.output_type,
        )
    
    # Rest stays the same...
```

## Phase 6: Remove Obsolete Code

### 6.1 Delete/Deprecate

1. **Delete entirely**:
   - `rehydrate_tool()` in tool_factory.py
   - Body comparison logic in tool resolution
   - Complex Tool validation in resolve_tools

2. **Mark as deprecated**:
   - `_get_registered_tool()` in agents.py
   - `instantiate_tool` parameters everywhere
   - Tool.body field (keep for now but don't use)

## Phase 7: Update Tests

### 7.1 Create comprehensive test suite

```python
# test_tool_registry_refactor.py

import pytest
from iointel.src.utilities.tool_registry_utils import resolve_tool
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env

def test_string_resolution():
    """Test simple string tool resolution."""
    tool = resolve_tool("run_shell_command")
    assert tool.name == "run_shell_command"
    assert tool.fn is not None

def test_dict_resolution():
    """Test dict tool resolution (from YAML)."""
    tool = resolve_tool({"name": "run_shell_command", "unused": "data"})
    assert tool.name == "run_shell_command"
    assert tool.fn is not None

def test_tuple_resolution():
    """Test tuple resolution for stateful tools."""
    tool = resolve_tool(("file_read", {"base_path": "/tmp"}))
    assert tool.name == "file_read"
    assert tool.fn is not None

def test_invalid_tool_name():
    """Test error on invalid tool name."""
    with pytest.raises(ValueError) as exc:
        resolve_tool("non_existent_tool")
    assert "not found in registry" in str(exc.value)
    assert "Available tools include:" in str(exc.value)

def test_yaml_round_trip():
    """Test workflow YAML serialization/deserialization."""
    from iointel.src.workflow import Workflow
    from iointel.src.agents import Agent
    
    # Create workflow
    workflow = Workflow(
        agents=[Agent(name="test", tools=["run_shell_command"])],
        tasks=[{"type": "agent", "agents": ["test"]}]
    )
    
    # Serialize
    yaml_str = workflow.to_yaml()
    
    # Deserialize
    restored = Workflow.from_yaml(yaml_str)
    
    # Verify tools work
    agent = restored.agents[0]
    assert len(agent.tools) == 1
    assert agent.tools[0].fn is not None
```

## Migration Checklist

- [ ] Update tool_registry_utils.py with simplified resolve_tool
- [ ] Simplify tool_factory.py to use centralized resolution
- [ ] Update agents.py to use centralized resolution
- [ ] Fix workflow.py YAML handling
- [ ] Update agents_factory.py
- [ ] Add Tool.to_yaml_ref() and from_yaml_ref()
- [ ] Create comprehensive test suite
- [ ] Test with web workflow execution
- [ ] Update documentation
- [ ] Remove deprecated code in next version

## Benefits

1. **Simplicity**: One way to resolve tools - registry lookup by name
2. **Reliability**: No more NoneType errors from missing fn
3. **Performance**: No rehydration or body comparison needed
4. **Maintainability**: Much less code, clearer flow
5. **Compatibility**: YAML files just have tool names, work anywhere

## Rollback Plan

If issues arise:
1. The refactor preserves backward compatibility where possible
2. Old parameters are ignored rather than causing errors
3. Can revert by keeping old functions temporarily as fallbacks
4. Gradual migration allows testing at each phase