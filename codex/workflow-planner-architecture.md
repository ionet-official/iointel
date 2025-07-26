# iointel Workflow Planner Application Architecture

## High-Level Overview

The iointel workflow planner is a sophisticated system that allows users to create, validate, and execute AI-powered workflows using natural language descriptions. It bridges three different ontological representations and provides multiple execution modes.

## Core Architecture Components

### 1. **Workflow Ontology Trinity**

The system operates with three distinct but interconnected workflow representations:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WorkflowSpec  │    │WorkflowDefinition│    │  Runtime Dict   │
│   (Modern DAG)  │◄──►│  (Legacy YAML)  │◄──►│  (add_task())   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • NodeSpec      │    │ • TaskDefinition│    │ • Plain dicts   │
│ • Required: id  │    │ • Required:     │    │ • Optional:     │
│ • React Flow    │    │   task_id       │    │   name, type    │
│ • LLM Generated │    │ • YAML export   │    │ • No validation │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. **Agent Result Format System**

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentResultFormat                        │
├─────────────────────────────────────────────────────────────┤
│ chat          │ ['result']                                  │
│ chat_w_tools  │ ['result', 'tool_usage_results']           │
│ workflow      │ ['result', 'conversation_id',              │
│               │  'tool_usage_results']                      │
│ full          │ ['result', 'conversation_id',              │
│               │  'tool_usage_results', 'full_result']      │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                               │
├─────────────────────────┬─────────────────────┬─────────────────────────────────┤
│    Natural Language     │    Gradio Web UI    │      Direct API Calls          │
│    Workflow Request     │    (workflow_server)│      (Workflow.add_task)       │
└─────────────────────────┴─────────────────────┴─────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            WORKFLOW PLANNING LAYER                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WorkflowPlanner Agent                                                          │
│  ├─ LLM-powered workflow generation                                             │
│  ├─ Tool catalog integration (create_tool_catalog)                             │
│  ├─ Parameter validation (required vs optional)                                │
│  └─ Multi-attempt validation with feedback loops                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ONTOLOGY CONVERSION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  dict_to_task_definition() ◄──► task_definition_to_dict()                      │
│  ├─ Schema validation (Pydantic)                                               │
│  ├─ Field normalization (task_id, name generation)                             │
│  └─ Backward compatibility preservation                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            WORKFLOW EXECUTION LAYER                             │
├─────────────────────────┬───────────────────────────────────────────────────────┤
│    DAG Execution        │           Sequential Execution                        │
│    (WorkflowSpec)       │           (TaskDefinition)                            │
├─────────────────────────┼───────────────────────────────────────────────────────┤
│ • DAGExecutor           │ • Workflow.execute_graph()                            │
│ • Parallel batches      │ • TaskNode chain execution                            │
│ • Topology resolution   │ • Linear task progression                             │
│ • Complex dependencies  │ • Simple workflows                                    │
└─────────────────────────┴───────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PARAMETER PROPAGATION CHAIN                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Workflow.run_tasks(agent_result_format)                                       │
│           │                                                                     │
│           ▼                                                                     │
│  execute_graph() ──► graph_nodes.run() ──► workflow.run_task()                 │
│           │                                      │                             │
│           ▼                                      ▼                             │
│  execute_agent_task() ──► run_agents() ──► Task.run() ──► Agent.run()          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AGENT EXECUTION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Agent Instances                                                                │
│  ├─ PydanticAgent (pydantic-ai integration)                                    │
│  ├─ Tool execution with parameter validation                                   │
│  ├─ Memory integration (AsyncMemory)                                           │
│  ├─ Conversation management                                                     │
│  └─ Result formatting (AgentResultFormat)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TOOL EXECUTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Tool Registry (TOOLS_REGISTRY)                                                │
│  ├─ Dynamic tool loading from environment                                      │
│  ├─ Function introspection (func_metadata)                                     │
│  ├─ Parameter validation with execution_metadata filtering                     │
│  └─ Async/sync execution support                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RESULT PROCESSING                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AgentResultFormat Filtering                                                   │
│  ├─ _postprocess_agent_result()                                                │
│  ├─ Field selection based on use case                                          │
│  ├─ Tool usage extraction                                                       │
│  └─ Memory storage (conversation history)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Parameterization Surfaces

### 1. **Workflow Creation Parameters**
```python
# Natural Language Interface
user_query: str  # "Create a workflow to analyze crypto prices"
tool_catalog: Dict[str, Any]  # Available tools and their schemas

# Direct Programming Interface  
Workflow(
    objective: str,
    client_mode: bool = True,
    agents: Optional[List[Agent]] = None
)
```

### 2. **Task Definition Parameters**
```python
# Runtime Dict Format (add_task)
{
    'type': Literal["agent", "tool", "workflow_call", "decision"],
    'name': str,  # Task identifier
    'objective': str,  # Task description
    'task_metadata': {
        'agent_instructions': str,  # Agent-specific instructions
        'config': Dict[str, Any],   # Tool configuration
        'available_results': Dict,  # Previous task results
    },
    'execution_metadata': {
        'conversation_id': str,
        'agent_result_format': Literal["chat", "chat_w_tools", "workflow", "full"]
    }
}
```

### 3. **Execution Parameters**
```python
# Workflow Execution
workflow.run_tasks(
    conversation_id: Optional[str] = None,
    agent_result_format: str = "full",  # Result filtering
    **kwargs
)

# Agent Execution
agent.run(
    query: str,
    conversation_id: Optional[str] = None,
    result_format: Optional[Union[AgentResultFormat, List[str]]] = None,
    message_history_limit: int = 100,
    **kwargs
)
```

### 4. **Tool Integration Parameters**
```python
# Tool Registration
@register_tool
def tool_function(
    param1: str,
    param2: Optional[int] = None,
    execution_metadata: Optional[Dict] = None  # Auto-filtered by introspection
):
    pass

# Tool Catalog Structure
{
    "tool_name": {
        "parameters": {"param1": "str", "param2": "int"},
        "required_parameters": ["param1"],  # Only actually required
        "description": str,
        "is_async": bool
    }
}
```

## Data Flow Architecture

```
User Request
     │
     ▼
[WorkflowPlanner] ──► WorkflowSpec (DAG nodes/edges)
     │                       │
     ▼                       ▼
[Validation] ◄──── [Tool Catalog Integration]
     │                       │
     ▼                       ▼
[Ontology Conversion] ──► TaskDefinition List
     │
     ▼
[Execution Router] 
     │
     ├─► [DAG Executor] (complex workflows)
     │        │
     │        ▼
     │   [Parallel Batches] ──► [Task Nodes]
     │
     └─► [Sequential Executor] (simple workflows)
              │
              ▼
         [TaskNode Chain] ──► [Agent Execution]
                                    │
                                    ▼
                               [Tool Calls] ──► [Results]
                                    │
                                    ▼
                            [Result Formatting] ──► Output
```

### Critical Discovery: Agent Data Access Architecture

**Key Insight**: The workflow system operates on **two distinct levels**:

1. **Graph Topology**: Controls execution order and dependencies via edges
2. **Data Context**: Provides agents with ALL available workflow data (not just connected nodes)

#### Agent Data Access Pattern
```
WorkflowState.results (ALL previous execution results)
    ↓
graph_nodes.py: Add to task_metadata["available_results"] 
    ↓
chainables.py: Extract and process for agent context
    ↓
Agent: Receives comprehensive context from entire workflow history
    ↓
Agent reasoning: Can reference ANY previous workflow data
```

#### Why This Design is Brilliant
- **Fault Tolerance**: Agents can work around failed tool nodes
- **Intelligence**: Full context enables better reasoning and decision-making
- **Autonomy**: Agents can use their own tools independent of workflow tool nodes
- **Resilience**: Multiple execution paths for the same functionality

## Configuration and Extensibility Points

### 1. **Environment Configuration**
- **Tool Loading**: `load_tools_from_env("creds.env")`
- **Model Configuration**: OpenAI API keys, base URLs
- **Memory Backends**: AsyncMemory implementations

### 2. **Agent Configuration**
- **Model Selection**: OpenAI models, custom providers
- **Tool Selection**: Per-agent tool subsets
- **Memory Management**: Conversation persistence
- **Output Types**: Structured vs unstructured responses

### 3. **Workflow Execution Modes**
- **Sequential**: Traditional task chains
- **DAG**: Complex dependency graphs with parallel execution
- **Streaming**: Real-time execution updates
- **Client Mode**: External service integration

### 4. **Result Format Customization**
- **Programmatic**: `AgentResultFormat.custom(fields=["result", "custom_field"])`
- **Use Case Specific**: Chat, tools, workflow chaining, debugging
- **Backward Compatible**: Legacy `include_fields` parameter support

## Key Innovation Points

1. **Ontology Bridging**: Seamless conversion between three workflow representations
2. **Parameter Introspection**: Automatic `execution_metadata` filtering based on function signatures  
3. **Semantic Result Formats**: Context-aware field selection for different use cases
4. **Validation Feedback Loops**: Multi-attempt workflow generation with error correction
5. **Hybrid Execution**: Automatic selection between sequential and DAG execution modes
6. **Tool Parameter Intelligence**: Proper distinction between required and optional parameters

## Implementation Details

### Workflow Ontology Conversion Flow

```python
# 1. Natural Language → WorkflowSpec
user_query = "Analyze Bitcoin prices and trends"
workflow_spec = workflow_planner.generate_workflow(user_query, tool_catalog)

# 2. WorkflowSpec → TaskDefinition (via conversion)
task_definitions = []
for node in workflow_spec.nodes:
    task_dict = {
        'type': node.type,
        'name': node.label,
        'task_metadata': {'config': node.data.config}
    }
    task_def = dict_to_task_definition(task_dict)
    task_definitions.append(task_def)

# 3. TaskDefinition → Runtime Dict (for execution)
for task_def in task_definitions:
    runtime_task = task_definition_to_dict(task_def)
    workflow.add_task(runtime_task)
```

### Parameter Propagation Chain

```python
# Complete parameter flow
result = await workflow.run_tasks(
    agent_result_format="chat"  # Entry point
)

# Internal propagation:
# workflow.py:run_tasks() 
#   → execute_graph(agent_result_format="chat")
#   → current_node._agent_result_format = "chat"
#   → graph_nodes.py:run()
#   → workflow.run_task(agent_result_format="chat")
#   → execution_metadata["agent_result_format"] = "chat"
#   → execute_agent_task(execution_metadata)
#   → AgentResultFormat.chat()
#   → agent.run(result_format=format_instance)
#   → _postprocess_agent_result(result_format=format_instance)
#   → return filtered_result
```

### Tool Parameter Validation Logic

```python
# Tool catalog creation with proper parameter extraction
def create_tool_catalog():
    for tool_name, tool in TOOLS_REGISTRY.items():
        # Extract from JSON schema
        properties = tool.parameters.get("properties", {})
        required_params = tool.parameters.get("required", [])
        
        # Handle complex anyOf types (nullable optionals)
        parameters = {}
        for param_name, param_info in properties.items():
            if "anyOf" in param_info:
                # Find non-null type
                for type_option in param_info["anyOf"]:
                    if type_option.get("type") != "null":
                        param_type = type_option.get("type", "any")
                        break
            else:
                param_type = param_info.get("type", "any")
            
            parameters[param_name] = param_type
        
        catalog[tool_name] = {
            "parameters": parameters,           # All parameters
            "required_parameters": required_params,  # Only required ones
        }

# Validation using corrected logic
def validate_tool_parameters(node, tool_catalog):
    required_params = tool_catalog[node.tool_name]["required_parameters"]
    config_params = set(node.config.keys())
    
    # Only check actually required parameters
    missing_params = set(required_params) - config_params
    if missing_params:
        return f"Missing required parameters: {missing_params}"
```

### AgentResultFormat Implementation

```python
class AgentResultFormat(BaseModel):
    include_result: bool = Field(True)
    include_conversation_id: bool = Field(False)
    include_tool_usage_results: bool = Field(False)
    include_full_result: bool = Field(False)
    
    @classmethod
    def chat(cls) -> "AgentResultFormat":
        return cls(include_result=True)  # Only result
    
    @classmethod
    def workflow(cls) -> "AgentResultFormat":
        return cls(
            include_result=True,
            include_conversation_id=True,
            include_tool_usage_results=True
        )  # For chaining
    
    def get_included_fields(self) -> List[str]:
        fields = []
        if self.include_result: fields.append('result')
        if self.include_conversation_id: fields.append('conversation_id')
        if self.include_tool_usage_results: fields.append('tool_usage_results')
        if self.include_full_result: fields.append('full_result')
        return fields
```

## File Structure Overview

```
iointel/src/
├── workflow.py                 # Main Workflow class, ontology conversion
├── agents.py                   # Agent execution, result formatting
├── chainables.py              # Task executors, AgentResultFormat integration
├── utilities/
│   ├── runners.py             # run_agents wrapper, parameter propagation
│   ├── graph_nodes.py         # TaskNode execution, state management
│   ├── func_metadata.py       # Function introspection, execution_metadata
│   └── dag_executor.py        # DAG topology execution
├── agent_methods/
│   ├── agents/
│   │   └── workflow_planner.py # LLM workflow generation
│   ├── data_models/
│   │   ├── datamodels.py      # AgentResultFormat, TaskDefinition
│   │   └── workflow_spec.py   # WorkflowSpec, NodeSpec validation
│   └── tools/                 # Tool implementations
├── web/
│   └── workflow_server.py     # Gradio UI, tool catalog creation
└── cli/
    └── run_workflow_planner.py # CLI interface, tool catalog
```

## Usage Examples

### 1. Natural Language Workflow Creation
```python
# Via Gradio Web UI
user_input = "Create a workflow to get Bitcoin price and analyze trends"
# → WorkflowPlanner generates WorkflowSpec
# → Validates against tool catalog  
# → Converts to executable format
# → Returns workflow results
```

### 2. Programmatic Workflow Creation
```python
workflow = Workflow(objective="Analyze crypto data")
workflow.add_task({
    'type': 'agent',
    'name': 'price_analyzer',
    'task_metadata': {
        'agent_instructions': 'Get Bitcoin current price and analyze trends'
    }
})
results = await workflow.run_tasks(agent_result_format="chat")
# Returns: {'results': {'price_analyzer': {'result': 'Bitcoin analysis...'}}}
```

### 3. Custom Result Formatting
```python
# Using semantic categories
chat_result = await workflow.run_tasks(agent_result_format="chat")
# {'results': {'task1': {'result': 'response'}}}

workflow_result = await workflow.run_tasks(agent_result_format="workflow") 
# {'results': {'task1': {'result': 'response', 'conversation_id': 'uuid', 'tool_usage_results': [...]}}}

# Using programmatic formatting
custom_format = AgentResultFormat(
    include_result=True,
    include_tool_usage_results=True,
    include_conversation_id=False
)
agent_result = await agent.run(query, result_format=custom_format)
```

This architecture enables natural language workflow creation while maintaining programmatic control and extensibility at every layer.