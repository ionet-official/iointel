# Workflow Data Flow Safety - Future Implementation TODO

## Problem Statement

The current "Maximum Available Context" architecture provides excellent fault tolerance and intelligence, but raises **critical concerns for production/safety workflows** where agents can bypass quality gates and safety controls.

## Production Safety Risks

### Factory/Manufacturing Risk:
```
sensor_reading → quality_check → FAILED → production_line_agent
```
- ❌ **Risk**: Agent receives raw sensor data despite failed quality check
- ❌ **Result**: Defective products shipped bypassing quality gates

### Trading System Risk:
```
market_data → risk_assessment → FAILED → trading_agent  
```
- ❌ **Risk**: Agent receives market data despite failed risk assessment
- ❌ **Result**: Massive financial losses ignoring risk controls

## TODO: Implement Role-Based Data Flow Control

### 1. Auto-Detection System
- [ ] **Agent Role Pattern Matching**
  ```python
  AGENT_ROLE_PATTERNS = {
      "quality_": "strict",      # Quality agents get edge-only data
      "inspector": "strict",     # Inspectors get edge-only data  
      "controller": "strict",    # Controllers get edge-only data
      "trader": "strict",        # Traders get edge-only data
      "analyst": "maximum",      # Analysts get full context
      "researcher": "maximum",   # Researchers get full context
  }
  ```

- [ ] **Tool-Based Safety Detection**
  ```python
  DANGEROUS_TOOLS = [
      "production_control", "trade_execution", "financial_transfer",
      "system_shutdown", "emergency_stop", "quality_gate"
  ]
  # Auto-trigger strict mode for dangerous tools
  ```

- [ ] **Workflow Context Inference**
  ```python
  # Detect safety-critical vs research workflows by keywords
  safety_keywords = ["production", "trading", "quality", "control", "safety"]
  research_keywords = ["analysis", "research", "exploration", "investigation"]
  ```

### 2. Data Flow Modes Implementation

- [ ] **Strict Mode** (Production/Safety)
  ```python
  # Agent receives only edge-connected data
  agent_context = {
      "immediate_inputs": filtered_by_edges_only,
      "execution_metadata": {...}  # Non-data context only
  }
  ```

- [ ] **Maximum Mode** (Research/Analysis) 
  ```python
  # Current behavior - full workflow context
  agent_context = {
      "available_results": all_previous_results,
      "workflow_history": {...}
  }
  ```

- [ ] **Hybrid Mode** (Best of Both)
  ```python
  # Separated immediate vs contextual data
  agent_context = {
      "immediate_inputs": filtered_by_edges,      # Primary task data
      "workflow_context": all_previous_results   # Read-only context
  }
  ```

### 3. LLM-Friendly Design

- [ ] **Simple Domain Instructions**
  ```
  For PRODUCTION/SAFETY workflows:
  - Use names like: quality_inspector, production_controller, risk_manager
  - These agents automatically respect safety controls and quality gates
  
  For RESEARCH/ANALYSIS workflows:
  - Use names like: data_analyst, research_agent, pattern_explorer  
  - These agents automatically access all available workflow data
  ```

- [ ] **No Technical Complexity for LLM**
  - Uses familiar domain concepts instead of technical abstractions
  - Leverages existing LLM knowledge about roles and responsibilities
  - Safety by default for production-sounding agent names

### 4. Implementation Strategy

#### Phase 1: Role Detection System
- [ ] **Create `detect_agent_role()` function**
  - Pattern match against safety vs research keywords
  - Analyze agent name, instructions, and tools
  - Return "strict", "maximum", or "hybrid"

- [ ] **Add role detection to workflow conversion**
  - Integrate into `workflow_converter.py`
  - Auto-assign data flow modes based on detected roles
  - Preserve explicit overrides when specified

#### Phase 2: Data Flow Filtering  
- [ ] **Modify `execute_agent_task()` in `chainables.py`**
  ```python
  data_flow_mode = detect_data_flow_mode(node, workflow)
  
  if data_flow_mode == "strict":
      context = filter_by_edges(available_results, node_edges)
  elif data_flow_mode == "maximum":
      context = {"available_results": available_results}  # Current behavior
  elif data_flow_mode == "hybrid":
      context = {
          "immediate_inputs": filter_by_edges(...),
          "workflow_context": available_results
      }
  ```

- [ ] **Create `filter_by_edges()` utility function**
  ```python
  def filter_by_edges(available_results: dict, node_edges: List[EdgeSpec]) -> dict:
      """Filter results to only include data from connected edges."""
      connected_sources = [edge.source for edge in node_edges if edge.target == node_id]
      return {k: v for k, v in available_results.items() if k in connected_sources}
  ```

#### Phase 3: Workflow-Level Policies
- [ ] **Add optional workflow-level data flow policy**
  ```python
  class WorkflowSpec(BaseModel):
      data_flow_policy: Optional[Literal["strict", "maximum", "hybrid"]] = None
  ```

- [ ] **Create policy resolution hierarchy**
  1. Explicit workflow policy overrides everything
  2. Auto-detect from agent role patterns  
  3. Fallback to safe defaults based on workflow context

#### Phase 4: Testing & Validation
- [ ] **Safety Gate Compliance Tests**
  - Production agents only receive approved data
  - Failed quality gates properly block downstream agents
  - Risk controls cannot be bypassed

- [ ] **Intelligence Preservation Tests**
  - Research workflows maintain analytical capabilities
  - Full context available for correlation and analysis
  - Fault tolerance preserved for appropriate workflows

- [ ] **Performance Impact Tests**
  - Minimal overhead from data flow filtering
  - Edge lookup performance optimization
  - Memory usage analysis for large workflows

### 5. Benefits

#### Production Safety
- ✅ Critical workflows get strict data flow control automatically
- ✅ Quality gates and safety controls cannot be bypassed
- ✅ Clear audit trails for compliance requirements

#### LLM Simplicity  
- ✅ No need to understand technical data flow modes
- ✅ Uses intuitive role-based naming patterns
- ✅ Leverages domain knowledge LLM already has

#### Flexible Implementation
- ✅ Explicit override available: `data_flow_mode="strict"`
- ✅ Progressive complexity disclosure
- ✅ Backward compatible with existing workflows

#### Smart Defaults
- ✅ Production workflows get safety controls automatically
- ✅ Research workflows get intelligence benefits automatically  
- ✅ Mixed workflows get hybrid approach

## Use Cases

### Strict Data Flow (Production/Safety)
- **Manufacturing**: Quality gates must be respected
- **Trading**: Risk controls cannot be bypassed
- **Medical**: Safety checks must block downstream actions
- **Compliance**: Audit trails must show proper data flow

### Maximum Context (Research/Analysis)
- **Research**: Benefit from all available data correlation
- **Debugging**: Need full workflow context for analysis
- **Intelligence**: Cross-reference multiple data sources
- **Exploration**: Discover patterns across unconnected data

### Hybrid Mode (Best of Both)
- **Complex Production**: Safety controls + intelligent context
- **Audit Compliance**: Strict primary data + full audit context
- **Risk Management**: Control gates + comprehensive analysis

## Priority & Impact

- **Priority**: High - Critical for production deployment safety
- **Complexity**: Medium - Requires careful role detection and data filtering  
- **Impact**: High - Enables safe production use while preserving research capabilities
- **Timeline**: Future implementation when production safety becomes critical

## Key Design Principles

1. **Safety by Default**: Production-sounding workflows automatically get strict controls
2. **Intelligence by Default**: Research-sounding workflows automatically get full context
3. **LLM Simplicity**: Use domain concepts, not technical abstractions
4. **Progressive Complexity**: Simple defaults with explicit override capability
5. **Backward Compatibility**: Existing workflows continue working unchanged

## Files to Modify

- [ ] `iointel/src/agent_methods/data_models/workflow_spec.py` - Add data flow mode support
- [ ] `iointel/src/workflow_converter.py` - Add role detection and mode assignment
- [ ] `iointel/src/chainables.py` - Implement data flow filtering in execute_agent_task()
- [ ] `iointel/src/agent_methods/agents/workflow_planner.py` - Update LLM instructions
- [ ] `iointel/src/utilities/data_flow_resolver.py` - Add edge-based filtering utilities

## Status

🎯 **DESIGN DOCUMENTED** - Ready for future implementation when production safety becomes critical

This addresses the fundamental production safety concern while maintaining the intelligent, fault-tolerant benefits of the current system through a role-based approach that's LLM-friendly and backward compatible.