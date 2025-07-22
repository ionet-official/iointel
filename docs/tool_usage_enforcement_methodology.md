# Tool Usage Enforcement Methodology
## Runtime SLA Enforcement as First-Class Workflow Primitive

### Core Philosophy

Tool usage enforcement is not just error handling - it's a **first-class runtime primitive** that ensures data flow integrity in agentic workflows. Each node type has inherent SLA requirements that must be satisfied before data can pass to downstream nodes.

---

## The Three Layers

### 1. **Product Layer** - Node Types with Built-in SLAs
Each node type is a product with implicit guarantees:

```python
decision_node: {
    "guarantee": "MUST route to exactly one path using decision tools",
    "sla": "conditional_gate tool usage required",
    "data_contract": "produces routed_to field"
}

data_node: {
    "guarantee": "MUST fetch current data from external sources", 
    "sla": "at least one data-fetching tool call required",
    "data_contract": "produces fresh data with timestamps"
}

executor_node: {
    "guarantee": "MUST perform requested action",
    "sla": "action tool usage required",
    "data_contract": "produces execution confirmation"
}
```

### 2. **Agentic Layer** - Tool Catalog for Reasoning
The WorkflowPlanner has access to a **decision tools catalog** that makes tool→SLA relationships explicit:

```python
DECISION_TOOLS_CATALOG = {
    "conditional_gate": {
        "effect": "routes workflow based on boolean condition",
        "requires_sla": True,
        "typical_patterns": ["if price > threshold", "if sentiment == positive"],
        "must_be_final_tool": True
    }
}
```

This allows the planner to **reason about enforcement** during workflow creation:
- "I need routing → I'll use conditional_gate → This creates SLA requirement"
- "User wants data analysis → I'll add get_stock_price → No SLA needed"

### 3. **Runtime Layer** - Validator Wrapper System
A **meta runtime helper** that wraps node execution with SLA validation:

```python
class NodeExecutionWrapper:
    async def execute_with_sla(self, node, input_data):
        # 1. Extract SLA requirements from node type/config
        sla_requirements = self.get_sla_requirements(node)
        
        # 2. Execute node with validation wrapper
        for attempt in range(max_retries):
            result = await self.execute_node(node, input_data)
            
            # 3. Validate SLA compliance
            if self.validate_sla(result, sla_requirements):
                return result  # Pass data downstream
            
            # 4. Block message passing, enhance prompt, retry
            input_data = self.enhance_for_retry(input_data, sla_requirements)
        
        # 5. Timeout/failure handling
        return self.handle_sla_failure(result, sla_requirements)
```

---

## Message Passing Control

The validator acts as a **message passing gatekeeper**:

```
[Previous Node] → [Data] → [Validator] → [Current Node] → [Validator] → [Next Node]
                            ↓                             ↓
                         "Is input valid?"           "Does output meet SLA?"
                            ↓                             ↓
                       [Allow/Block]                [Allow/Block/Retry]
```

**Benefits:**
- **Data Flow Integrity**: Malformed data can't propagate
- **SLA Guarantee**: Each node delivers on its promises  
- **Debugging**: Clear point of failure isolation
- **Composability**: Mix-and-match nodes with confidence

---

## Implementation Strategy

### Phase 1: Tool Catalog Integration
```python
# Add to WorkflowPlanner system prompt
DECISION_TOOLS_KNOWLEDGE = """
Available Decision Tools:
- conditional_gate: Routes based on boolean condition (SLA: must be final tool)
- boolean_gate: Simple true/false routing

When using decision tools, the agent MUST use them to route the workflow.
Example: Stock Decision Agent → include conditional_gate → creates SLA requirement
"""
```

### Phase 2: Node-Level SLA Specification
```python
class NodeSpec(BaseModel):
    # ... existing fields
    sla_requirements: Optional[NodeSLA] = None
    
class NodeSLA(BaseModel):
    tool_usage_required: bool
    required_final_tool: Optional[str] = None
    min_tool_calls: int = 0
    timeout_seconds: int = 120
```

### Phase 3: Runtime Wrapper Integration
```python
# In DAG executor
async def _execute_node_with_sla(self, node_id: str, state: WorkflowState):
    node = self.nodes[node_id]
    wrapper = NodeExecutionWrapper(node.sla_requirements)
    
    result = await wrapper.execute_with_sla(
        node_executor=lambda: self._execute_node(node_id, state),
        input_data=state.get_node_inputs(node_id)
    )
    
    return result
```

### Phase 4: Feedback Loop
```python
# SLA failures feed back to planner for learning
class WorkflowPlanner:
    def add_sla_failure_context(self, failure_info):
        self.context += f"""
        Previous SLA Failure: {failure_info.node_type} node failed because {failure_info.reason}
        Avoid this by ensuring proper tool usage requirements.
        """
```

---

## Example: Stock Decision Node

**Planning Time:**
```python
# WorkflowPlanner sees: "decide whether to buy/sell stock"
# Reasoning: "Need routing → use conditional_gate → creates SLA"
node = {
    "type": "agent",
    "label": "Stock Decision Agent", 
    "agent_instructions": "Analyze stock and route to buy/sell path",
    "tools": ["get_current_stock_price", "conditional_gate"],
    "sla_requirements": {
        "tool_usage_required": True,
        "required_final_tool": "conditional_gate",
        "min_tool_calls": 1
    }
}
```

**Runtime:**
```python
# Execution wrapper ensures SLA compliance
attempt_1: agent provides analysis text → SLA FAIL (no tools) → retry
attempt_2: agent calls get_stock_price only → SLA FAIL (no routing) → retry  
attempt_3: agent calls conditional_gate → SLA PASS → data flows downstream
```

---

## Benefits of This Methodology

1. **Product Clarity**: Each node type has clear contracts
2. **Agentic Reasoning**: Planner understands tool→SLA relationships
3. **Runtime Reliability**: SLA enforcement ensures data integrity
4. **Composability**: Mix nodes with confidence in their guarantees
5. **Debugging**: Clear failure points and retry logic
6. **Scalability**: Add new node types with their own SLAs

This creates a **self-documenting, self-enforcing workflow system** where the product (node types), planning (agentic reasoning), and execution (runtime validation) are aligned around explicit SLA contracts.