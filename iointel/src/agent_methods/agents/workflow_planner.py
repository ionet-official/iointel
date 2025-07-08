import os
import uuid
from typing import Dict, Any, Optional, List
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

🔧 Tool Usage Guidelines
------------------------
For each tool in the catalog, you'll see:
- `name`: The exact tool identifier to use in `tool_name`
- `description`: What the tool does
- `parameters`: Required configuration keys

When creating a tool node:
1. Set `type` to "tool"
2. Set `data.tool_name` to the exact tool name from catalog
3. Include ALL required parameters in `data.config`
4. Map parameters correctly (e.g., if tool expects "url", don't use "endpoint")

⚡ Conditional Logic
--------------------
NEVER use string conditions in edges. Instead use explicit decision nodes:

1. **For simple comparisons**: Use decision tools
   - `json_evaluator`: Check JSON data conditions
   - `number_compare`: Compare numbers (>, <, ==)
   - `string_contains`: Check string patterns
   
2. **For complex logic**: Use agent nodes with decision instructions
   - Return structured decisions with reasoning
   - Output clear routing information

3. **For routing**: Use routing tools
   - `conditional_router`: Route based on structured decisions
   - `boolean_mux`: Simple true/false routing

Example: Instead of edge condition `"'rain' in weather_data"`, create:
```json
{
  "id": "check_rain", 
  "type": "decision",
  "data": {"tool_name": "json_evaluator", "config": {"expression": "data.weather.includes('rain')"}}
}
```

🎯 Output Requirements
----------------------
Generate a WorkflowSpec with:
- `title`: Clear, action-oriented name
- `description`: One sentence explaining the workflow purpose
- `nodes`: Array of nodes accomplishing the goal
- `edges`: Connections between nodes with optional conditions

Remember:
- Start simple - users can refine
- Each node needs a unique, descriptive ID
- Verify all tool names exist in the catalog
- Ensure data flows logically through the DAG
- NO authentication/security config - system handles that
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

Context: {full_context}

Generate a WorkflowSpec that fulfills the user's requirements using the available tools.
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
            
        return workflow_spec
    
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