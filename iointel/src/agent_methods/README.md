# Agent Methods - Workflow System Architecture

## Overview

This module implements the core workflow planning and execution system for the iointel agent framework. It provides a comprehensive architecture for creating, validating, and executing complex multi-agent workflows with proper separation of concerns between tools and data sources.

## Architecture Components

### 🤖 Agents (`agents/`)
- **WorkflowPlanner**: Specialized agent for generating React Flow compatible workflow specifications
- **AgentsFactory**: Factory pattern for agent instantiation and configuration
- **WorkflowPrompts**: Dynamic prompt generation with validation context

### 📊 Data Models (`data_models/`)
- **WorkflowSpec**: Core workflow specification with comprehensive validation
- **DataModels**: Pydantic models for structured data handling
- **DataSourceRegistry**: Registry pattern for data source management
- **TestAlignment**: Bidirectional workflow-test validation metadata

### 🔧 Tools (`tools/`)
- **Tool Registry**: Decorator-based tool registration system

## Key Architectural Decisions

### Tools vs Data Sources Separation

**Critical Design Principle**: Complete separation between tools and data sources to prevent architectural confusion.

```python
# Tools: Used by agents for computation/reasoning
agent_node = {
    "type": "agent",
    "data": {
        "agent_instructions": "Analyze the data",
        "tools": ["searxng_search", "calculator"]  # Tools for agent use
    }
}

# Data Sources: Input/trigger nodes (completely separate)
data_source_node = {
    "type": "data_source", 
    "data": {
        "source_name": "user_input",  # NOT in tools catalog
        "config": {"prompt": "Enter your query"}
    }
}
```

**Why This Matters:**
- Prevents "MISSING PARAMETERS" errors for data source nodes
- Clear conceptual boundary between input triggers and agent capabilities
- Enables proper validation and error messaging

### Workflow-Test Alignment System

Implements bidirectional metadata linking between workflows and test validation:

```python
class TestAlignment(BaseModel):
    test_ids: Set[str] = Field(default_factory=set)
    test_results: List[TestResult] = Field(default_factory=list)
    validation_status: Literal["untested", "passing", "failing", "mixed"] = "untested"
    production_ready: bool = Field(default=False)
```

**Goal**: Only fully validated workflows appear in production interface.

### Single Source of Truth Pattern

All workflow representations use standardized conversion:

```python
# Standard workflow-to-text conversion
workflow_text = workflow_spec.to_llm_prompt()

# Standard serialization
yaml_output = workflow_spec.to_yaml()
executable_format = workflow_spec.to_workflow_definition()
```

Prevents inconsistencies across different system components.

## Validation Architecture

### Multi-Layer Validation System

1. **Structural Validation**: Node/edge consistency, required fields
2. **Tool Validation**: Verify all referenced tools exist in catalog  
3. **Data Source Validation**: Confirm data source parameters are complete
4. **Routing Validation**: Decision nodes have proper routing configuration
5. **SLA Validation**: Service level agreement compatibility

### Detailed Feedback System

```python
# Provides specific remediation guidance
validation_errors = [
    "🔧 FIX: Add route_index to edges from decision nodes",
    "🔧 FIX: Use only valid tool names from provided catalog", 
    "🔧 FIX: Decision nodes MUST have outgoing edges for routing"
]
```

### Auto-Retry with Feedback

Workflow generation automatically retries with validation feedback:
- Max 3 attempts with detailed error context
- Previous workflow context for refinement
- Comprehensive logging for debugging

## Integration Points

### Web Interface Integration
- **Conversation Management**: Session versioning and context isolation
- **Search Integration**: Semantic search across workflows, tools, tests
- **Analytics Dashboard**: Test discovery and validation status

### Test System Integration  
- **Unified Test Repository**: Centralized test storage with categorization
- **Four-Layer Testing**: LOGICAL → AGENTIC → ORCHESTRATION → FEEDBACK
- **Tag-Based Discovery**: Flexible test filtering and execution

### Tool System Integration
- **Decorator Registration**: `@register_tool("tool_name")` pattern
- **OpenAI Compliance**: Function names follow `^[a-zA-Z0-9_-]+$` pattern
- **Dynamic Catalog**: Runtime tool discovery and validation

## Usage Patterns

### Creating a Workflow Planner

```python
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner

planner = WorkflowPlanner(
    conversation_id="workflow_session_001",
    model="gpt-4o",
    debug=True
)

workflow = await planner.generate_workflow(
    query="Create a stock analysis and trading workflow",
    tool_catalog=get_available_tools(),
    context={"trading_rules": "5% threshold for buy/sell"},
    max_retries=3
)
```

### Workflow Validation

```python
# Comprehensive validation with detailed feedback
validation_issues = workflow_spec.validate_structure(tool_catalog)

if validation_issues:
    for issue in validation_issues:
        print(f"⚠️ {issue}")
else:
    print("✅ Workflow validation passed")
```

### Test Integration

```python
# Link workflow to test validation
workflow_spec.test_alignment.test_ids.add("routing_test_001")
workflow_spec.test_alignment.validation_status = "passing"
workflow_spec.test_alignment.production_ready = True
```

## Security Considerations

- **Credential Protection**: All API keys excluded from git via enhanced .gitignore
- **Data Isolation**: Conversations, logs, and generated data not committed
- **Input Validation**: Comprehensive validation prevents injection attacks
- **Access Control**: Proper separation between system and user data

## Development Guidelines

1. **Type Everything**: All inputs/outputs must have proper type annotations
2. **Single Source of Truth**: Use standard conversion utilities consistently  
3. **Test Integration**: All workflows must integrate with unified test system
4. **Error Recovery**: Implement proper retry logic with feedback incorporation
5. **Documentation**: Update this README when making architectural changes

---

*This documentation reflects the current architecture after the July 26, 2025 consolidation and security improvements.*