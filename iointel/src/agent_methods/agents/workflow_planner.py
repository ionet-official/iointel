import os
import uuid
from typing import Dict, Any, Optional
from ...agents import Agent
from ...memory import AsyncMemory
from ..data_models.workflow_spec import WorkflowSpec


WORKFLOW_PLANNER_INSTRUCTIONS = """
You are WorkflowPlanner-GPT for IO.net, a specialized AI that designs executable workflows.

📌 Core Responsibility
----------------------
Transform user requirements into a structured workflow (DAG) using available tools and agents.
You output ONLY valid JSON conforming to WorkflowSpec schema - no explanations or comments.

🏗️ Workflow Taxonomy
--------------------

### Node Types (exactly one of these):

1. **tool** - Executes a specific tool from the catalog
   ```json
   {
     "id": "fetch_weather",
     "type": "tool",
     "label": "Get Weather Data",
     "data": {
       "tool_name": "weather_api",
       "config": {"location": "London", "units": "celsius"},
       "ins": [],
       "outs": ["weather_data", "status"]
     }
   }
   ```

2. **agent** - Runs an AI agent with instructions
   ```json
   {
     "id": "analyze_data",
     "type": "agent", 
     "label": "Analyze Results",
     "data": {
       "agent_instructions": "Analyze the weather patterns and summarize key insights",
       "config": {"model": "gpt-4", "temperature": 0.7},
       "ins": ["weather_data"],
       "outs": ["analysis", "insights"]
     }
   }
   ```

3. **decision** - Makes boolean or routing decisions (use decision tools or agents)
   ```json
   {
     "id": "check_rain",
     "type": "decision",
     "label": "Check Rain Condition", 
     "data": {
       "tool_name": "json_evaluator",
       "config": {"expression": "data.weather.condition == 'rain'"},
       "ins": ["weather_data"],
       "outs": ["result", "details"]
     }
   }
   ```

4. **workflow_call** - Executes another workflow
   ```json
   {
     "id": "run_sub_workflow",
     "type": "workflow_call",
     "label": "Process Subset",
     "data": {
       "workflow_id": "data_processing_v2",
       "config": {"mode": "batch"},
       "ins": ["raw_data"],
       "outs": ["processed_data"]
     }
   }
   ```

📊 Data Flow Rules
------------------
1. **Port Naming**: Use clear, semantic names
   - Inputs: data, query, config, source, input, params
   - Outputs: result, output, response, error, status

2. **Edge Connections**: Connect compatible ports
   ```json
   {
     "id": "edge1",
     "source": "fetch_data",
     "target": "process_data",
     "sourceHandle": "result",
     "targetHandle": "input"
   }
   ```

3. **Data References**: Use {node_id.port} syntax in config
   ```json
   "config": {
     "message": "Weather is {fetch_weather.result}",
     "data": "{analyze_data.insights}"
   }
   ```

🔧 CRITICAL TOOL USAGE RULES
----------------------------
⚠️  **ABSOLUTE REQUIREMENT**: You MUST ONLY use tools from the provided tool_catalog.
⚠️  **NEVER HALLUCINATE TOOLS**: Do not invent or assume tool names.
⚠️  **VALIDATION**: Every tool_name MUST exist in the catalog or the workflow will fail.

**Tool Catalog Format:**
```
tool_catalog = {
    "tool_name": {
        "name": "exact_tool_name",
        "description": "what the tool does",
        "parameters": {"param1": "type", "param2": "type"},
        "is_async": true/false
    }
}
```

**Mandatory Process for Tool Nodes:**
1. **CHECK CATALOG**: Verify tool exists in provided tool_catalog
2. **EXACT NAME**: Use the exact `name` field from catalog as `tool_name`
3. **REQUIRED PARAMS**: Include ALL parameters from catalog in `data.config`
4. **NO ASSUMPTIONS**: Don't assume similar tools exist

**If Required Tools Missing:**
- Use the `reasoning` field to explain what tools are needed
- Example: "Cannot create weather workflow - requires web search or weather api tool which is currentlynot available"
- Suggest alternative approaches using available tools

⚡ Conditional Logic
--------------------
🚫 FORBIDDEN: String conditions in edges (e.g., `"condition": "temperature < 65"`)
✅ REQUIRED: Use explicit decision nodes with proper tool configurations

**Pattern for conditional workflows:**
1. **Decision Node**: Use decision tools to evaluate conditions
2. **Router Node**: Use routing tools to direct flow based on decision results  
3. **Action Nodes**: Execute different actions based on routing

**Example: Mathematical calculation routing (using available tools)**
```json
{
  "nodes": [
    {"id": "calc_numbers", "type": "tool", "data": {"tool_name": "add", "config": {"a": 10, "b": 5}, "outs": ["result"]}},
    {"id": "check_result", "type": "decision", "data": {
      "tool_name": "number_compare",
      "config": {"operator": ">", "threshold": 10},
      "ins": ["result"], 
      "outs": ["is_greater", "details"]
    }},
    {"id": "route_action", "type": "decision", "data": {
      "tool_name": "conditional_router", 
      "config": {"routes": {"true": "multiply_action", "false": "divide_action"}},
      "ins": ["is_greater"],
      "outs": ["routed_to"]
    }},
    {"id": "multiply_action", "type": "tool", "data": {"tool_name": "multiply", "config": {"a": "{calc_numbers.result}", "b": 2}}},
    {"id": "divide_action", "type": "tool", "data": {"tool_name": "divide", "config": {"a": "{calc_numbers.result}", "b": 2}}}
  ],
  "edges": [
    {"source": "calc_numbers", "target": "check_result", "sourceHandle": "result", "targetHandle": "result"},
    {"source": "check_result", "target": "route_action", "sourceHandle": "is_greater", "targetHandle": "is_greater"}, 
    {"source": "route_action", "target": "multiply_action", "sourceHandle": "routed_to", "targetHandle": null},
    {"source": "route_action", "target": "divide_action", "sourceHandle": "routed_to", "targetHandle": null}
  ]
}
```

⚠️  **IMPORTANT**: This example uses tools like `add`, `multiply`, `divide` - verify these exist in your tool_catalog!

🔑 Key Rules:
- Decision nodes output structured data (true/false, route names)
- Router nodes consume decision data and output routing information
- Edges are NEVER conditional - they just carry data
- Use `conditional_router` or `boolean_mux` for path selection

🎯 Output Requirements
----------------------
Generate a WorkflowSpec with:
- `title`: Clear, action-oriented name
- `description`: One sentence explaining the workflow purpose
- `nodes`: Array of nodes accomplishing the goal
- `edges`: Connections between nodes with optional conditions
- `reasoning`: Your thought process including:
  - Which tools from the catalog you used and why
  - Any limitations or constraints you encountered
  - Alternative approaches if some tools are missing
  - Explanation of workflow logic and flow

🔍 CRITICAL VALIDATION RULES
----------------------------
EVERY workflow MUST pass these checks:

1. **Node ID Uniqueness**: Each node.id must be unique within the workflow
2. **Edge Validity**: Every edge.source and edge.target MUST reference existing node IDs
3. **Port Consistency**: sourceHandle and targetHandle should match node ins/outs
4. **🚨 TOOL EXISTENCE**: All tool_name values MUST exist in the provided tool catalog
5. **No Orphaned Nodes**: Every node should be connected (except start/end nodes)

🚨 **BEFORE GENERATING WORKFLOW**: 
- List all tools you plan to use
- Verify each tool exists in the provided tool_catalog
- If any required tool is missing, explain in reasoning field
- Suggest alternative approaches using available tools

When refining workflows, preserve the existing node structure and only add/modify as needed.
NEVER reference nodes that don't exist in the nodes array.

🎯 **FINAL CHECKLIST**:
- ✅ Every tool_name exists in tool_catalog
- ✅ All node IDs are unique
- ✅ All edges reference existing nodes
- ✅ Reasoning field explains tool choices and limitations
- ✅ No authentication/security config (system handles that)

⚠️  **TOOL HALLUCINATION = WORKFLOW FAILURE**
If you use a tool that doesn't exist in the catalog, the entire workflow will fail during execution.
"""


class WorkflowPlanner:
    """
    A specialized agent that generates React Flow compatible workflow specifications
    from user requirements.
    """
    
    def __init__(
        self,
        memory: Optional[AsyncMemory] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the WorkflowPlanner agent.
        
        Args:
            memory: Optional AsyncMemory instance for conversation history
            conversation_id: Optional conversation ID for memory tracking
            model: Model name (defaults to env MODEL_NAME or "gpt-4o")
            api_key: API key (defaults to env OPENAI_API_KEY)
            base_url: Base URL (defaults to env OPENAI_API_BASE)
            debug: Enable debug mode
            **kwargs: Additional arguments passed to Agent
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        
        # Initialize the underlying agent with structured output
        self.agent = Agent(
            name="WorkflowPlanner",
            instructions=WORKFLOW_PLANNER_INSTRUCTIONS,
            model=model or os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            memory=memory,
            conversation_id=self.conversation_id,
            output_type=WorkflowSpec,  # 🔑 guarantees structured JSON
            show_tool_calls=True,
            tool_pil_layout="horizontal",
            debug=debug,
            **kwargs
        )
    
    async def generate_workflow(
        self,
        query: str,
        tool_catalog: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> WorkflowSpec:
        """
        Generate a workflow specification from a user query.
        
        Args:
            query: User's natural language description of the workflow
            tool_catalog: Available tools and their specifications
            context: Additional context for workflow generation
            **kwargs: Additional arguments passed to agent.run()
            
        Returns:
            WorkflowSpec: Validated workflow specification ready for execution
        """
        # Prepare context with tool catalog
        full_context = {
            "tool_catalog": tool_catalog or {},
            "additional_context": context or {}
        }
        
        # Format the query with context
        formatted_query = f"""
User Query: {query}

🔧 AVAILABLE TOOLS IN CATALOG:
{self._format_tool_catalog(tool_catalog or {})}

🚨 CRITICAL REQUIREMENTS:
- You MUST ONLY use tools from the above catalog
- Use the EXACT tool names as listed
- If required tools are missing, explain in the reasoning field
- DO NOT hallucinate or invent tool names

Additional Context: {context or {}}

Generate a WorkflowSpec that fulfills the user's requirements using ONLY the available tools above.
"""
        
        # Run the agent to generate the workflow with limited message history
        result = await self.agent.run(
            formatted_query,
            conversation_id=self.conversation_id,
            message_history_limit=5,  # Limit to last 5 messages to prevent context overflow
            **kwargs
        )
        
        # Extract the structured output
        workflow_spec = result.get("result")
        if not isinstance(workflow_spec, WorkflowSpec):
            raise ValueError(f"Expected WorkflowSpec, got {type(workflow_spec)}")
        
        # 🚨 CRITICAL VALIDATION: Check for tool hallucination
        tool_issues = workflow_spec.validate_tools(tool_catalog or {})
        if tool_issues:
            error_msg = f"🚨 TOOL HALLUCINATION DETECTED:\n" + "\n".join(tool_issues)
            error_msg += f"\n\nWorkflow reasoning: {workflow_spec.reasoning}"
            raise ValueError(error_msg)
            
        return workflow_spec
    
    def _format_tool_catalog(self, tool_catalog: dict) -> str:
        """Format the tool catalog for clear display to the LLM."""
        if not tool_catalog:
            return "❌ NO TOOLS AVAILABLE - Cannot create workflow without tools"
        
        formatted_tools = []
        for tool_name, tool_info in tool_catalog.items():
            formatted_tools.append(f"""
📦 {tool_name}
   Description: {tool_info.get('description', 'No description')}
   Parameters: {tool_info.get('parameters', {})}
   Usage: {{"tool_name": "{tool_name}", "config": {{ ... }} }}
""")
        
        return f"""
Available Tools ({len(tool_catalog)} total):
{''.join(formatted_tools)}

🚨 REMINDER: Use ONLY these exact tool names. Any other tool will cause failure.
"""
    
    async def refine_workflow(
        self,
        workflow_spec: WorkflowSpec,
        feedback: str,
        **kwargs
    ) -> WorkflowSpec:
        """
        Refine an existing workflow based on user feedback.
        
        Args:
            workflow_spec: Current workflow specification
            feedback: User feedback for refinement
            **kwargs: Additional arguments passed to agent.run()
            
        Returns:
            WorkflowSpec: Refined workflow specification
        """
        refinement_query = f"""
Current Workflow: {workflow_spec.model_dump_json(indent=2)}

User Feedback: {feedback}

Please generate an improved WorkflowSpec that addresses the feedback while maintaining the core functionality.
"""
        
        result = await self.agent.run(
            refinement_query,
            conversation_id=self.conversation_id,
            message_history_limit=5,  # Limit to last 5 messages to prevent context overflow
            **kwargs
        )
        
        refined_spec = result.get("result")
        if not isinstance(refined_spec, WorkflowSpec):
            raise ValueError(f"Expected WorkflowSpec, got {type(refined_spec)}")
            
        return refined_spec
    
    def create_example_workflow(self, title: str = "Example Workflow") -> WorkflowSpec:
        """
        Create a simple example workflow for testing.
        
        Args:
            title: Title for the example workflow
            
        Returns:
            WorkflowSpec: Example workflow specification
        """
        from ..data_models.workflow_spec import NodeSpec, NodeData, EdgeSpec, EdgeData
        
        workflow_id = uuid.uuid4()
        
        # Create example nodes
        nodes = [
            NodeSpec(
                id="start",
                type="tool",
                label="Fetch Data",
                data=NodeData(
                    config={"source": "api", "endpoint": "/data"},
                    ins=["trigger"],
                    outs=["data", "status"]
                ),
                runtime={"timeout": 30}
            ),
            NodeSpec(
                id="process",
                type="agent",
                label="Process Data",
                data=NodeData(
                    config={"instructions": "Process the input data"},
                    ins=["data"],
                    outs=["result", "summary"]
                ),
                runtime={"timeout": 60}
            ),
            NodeSpec(
                id="output",
                type="tool",
                label="Save Result",
                data=NodeData(
                    config={"destination": "file", "format": "json"},
                    ins=["result"],
                    outs=["success"]
                ),
                runtime={"timeout": 15}
            )
        ]
        
        # Create edges
        edges = [
            EdgeSpec(
                id="start_to_process",
                source="start",
                target="process",
                sourceHandle="data",
                targetHandle="data",
                data=EdgeData(condition="status == 'success'")
            ),
            EdgeSpec(
                id="process_to_output",
                source="process",
                target="output",
                sourceHandle="result",
                targetHandle="result"
            )
        ]
        
        return WorkflowSpec(
            id=workflow_id,
            rev=1,
            title=title,
            description="A simple example workflow that fetches, processes, and saves data",
            nodes=nodes,
            edges=edges,
            metadata={
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["example", "demo"]
            }
        )