# IO.net Routing Ontology - First Order System

## The Problem
- **Tools output**: `routed_to = 'buy_path'`
- **LLM generates**: `decision == 'buy'`  
- **DAG executor**: `routed_to == 'buy_path'`
- **Result**: Complete mismatch → incorrect routing

## The Solution: Unified Routing Contract

### 1. Routing Tools Standard Output
All routing tools MUST output:
```python
GateResult(
    routed_to='buy',      # ✅ Simple, semantic route name
    action='branch',      # branch|continue|terminate  
    confidence=1.0,
    audit_trail={...}
)
```

**Not**: `'buy_path'`, `'buy_signal'` - just `'buy'`

### 2. LLM Edge Condition Standard
LLM MUST generate edge conditions as:
```json
{
  "condition": "routed_to == 'buy'"
}
```

**Not**: `"decision == 'buy'"` or `"action == 'buy'"`

### 3. DAG Executor Standard
DAG executor looks for ONLY:
```python
if "routed_to ==" in condition:
    # Extract expected route and match
```

### 4. SLA Integration
```json
{
  "sla": {
    "final_tool_must_be": "percentage_change_gate",
    "enforce_routing": true  // NEW: Ensure routing result propagates
  }
}
```

## Implementation Changes Needed

### A. Fix routing tool outputs
- `percentage_change_gate`: Return `'buy'/'sell'` not `'buy_path'/'sell_path'`
- `conditional_gate`: Return semantic names
- `threshold_gate`: Return semantic names

### B. LLM Prompt Engineering
Add to WorkflowPlanner prompt:
```
ROUTING RULES:
- When using routing tools (conditional_gate, percentage_change_gate), 
  edge conditions MUST use: routed_to == 'route_name'
- Never use: decision == 'X' or action == 'X' 
- Route names are: buy, sell, hold, terminate, etc.
```

### C. Validation Enhancement
Add validation rule:
```python
if has_routing_tools and "decision ==" in condition:
    issues.append("❌ Use 'routed_to ==' not 'decision =='"
```

This creates a **first-order system** where:
1. ✅ Routing tools output consistent format
2. ✅ LLM generates matching conditions  
3. ✅ DAG executor uses single pattern
4. ✅ SLA enforces routing tool usage
5. ✅ Failed routes block downstream execution