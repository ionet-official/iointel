# Typed Data Flow Architecture (In Progress)

## Current Problem

We have untyped dict passing throughout the workflow system, leading to:
- Brittle data flow with hardcoded field extraction
- Mixed concerns between storage, agent views, and frontend display  
- LLM doesn't understand valid connections without strong typing
- No compile-time validation of data flow compatibility

## Existing Models to Leverage

### WorkflowSpec Models (Already Have!)
```python
# From workflow_spec.py
@dataclass
class NodeData:
    """Already has the structure we need!"""
    ins: List[str] = field(default_factory=list)  # Input ports
    outs: List[str] = field(default_factory=list)  # Output ports
    config: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    agent_instructions: Optional[str] = None

@dataclass  
class EdgeData:
    """Already tracks data flow connections!"""
    condition: Optional[str] = None
    # sourceHandle and targetHandle in EdgeSpec map to port names
```

### What's Missing: Type Information

We need to enhance NodeData with type contracts:

```python
# Extend NodeData to include port types
@dataclass
class TypedNodeData(NodeData):
    """Enhance existing NodeData with type information"""
    input_types: Dict[str, Type] = field(default_factory=dict)  # {"price": float}
    output_types: Dict[str, Type] = field(default_factory=dict) # {"result": str}
```

## Implementation Plan

### 1. Use Existing EdgeSpec for Type Validation
```python
# EdgeSpec already has sourceHandle/targetHandle!
edge = EdgeSpec(
    source="bitcoin_price",
    sourceHandle="price",  # Output port name
    target="analyzer", 
    targetHandle="market_price"  # Input port name
)

# Add validation using existing structure
def validate_edge(edge: EdgeSpec, nodes: Dict[str, NodeSpec]) -> bool:
    source_node = nodes[edge.source]
    target_node = nodes[edge.target]
    
    # Get types from enhanced NodeData
    source_type = source_node.data.output_types.get(edge.sourceHandle)
    target_type = target_node.data.input_types.get(edge.targetHandle)
    
    return is_type_compatible(source_type, target_type)
```

### 2. Enhance Tool Catalog Generation
```python
# In create_tool_catalog() - extract type info from pydantic schemas
tool_info = {
    "name": tool.name,
    "parameters": parameters,  # Already extracted
    "input_types": extract_types_from_schema(tool.parameters),
    "output_type": extract_return_type(tool.get_wrapped_fn())
}
```

### 3. Update WorkflowPlanner Prompts
```prompt
When creating edges, use sourceHandle and targetHandle to specify ports:
- sourceHandle: The output port name from source node (from outs list)
- targetHandle: The input port name on target node (from ins list)

Example:
Node A: outs=["price", "timestamp"]  
Node B: ins=["market_price", "time"]
Edge: source=A, sourceHandle="price", target=B, targetHandle="market_price"
```

### 4. Data Flow Resolution with Types
```python
# Enhanced data_flow_resolver.py
def resolve_typed_value(edge: EdgeSpec, results: Dict[str, Any]) -> Any:
    """Resolve using existing edge port information"""
    source_result = results[edge.source]
    
    # Use sourceHandle to extract specific output
    if isinstance(source_result, dict) and edge.sourceHandle:
        return source_result.get(edge.sourceHandle)
    return source_result
```

## Benefits of Using Existing Models

1. **No new models needed** - NodeData, EdgeSpec already have the structure
2. **Backward compatible** - Just adding optional type fields
3. **LLM already understands** - ins/outs/sourceHandle/targetHandle  
4. **Validation ready** - EdgeSpec has all connection info

## Migration Path

1. Start with type hints in tool catalog
2. Enhance NodeData with input_types/output_types
3. Update data_flow_resolver to use EdgeSpec.sourceHandle/targetHandle
4. Add validation layer for type checking
5. Update LLM prompts to specify ports explicitly

## Next Steps

- [ ] Add type extraction to tool catalog generation
- [ ] Enhance NodeData with optional type fields
- [ ] Update data_flow_resolver to use port names
- [ ] Add edge validation for type compatibility
- [ ] Test with complex workflows