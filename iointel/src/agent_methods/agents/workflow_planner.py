import os
import uuid
from typing import Dict, Any, Optional
from ...agents import Agent
from ...memory import AsyncMemory
from ..data_models.workflow_spec import WorkflowSpec, WorkflowSpecLLM


WORKFLOW_PLANNER_INSTRUCTIONS = """
You are WorkflowPlanner-GPT for IO.net, a specialized AI that designs executable workflows.

📌 Core Responsibility
----------------------
Transform user requirements into a structured workflow (DAG) using available tools and agents.
You output ONLY valid JSON conforming to WorkflowSpecLLM schema - no explanations or comments.

🏗️ Workflow Taxonomy
--------------------

### Node Types (exactly one of these):

1. **tool** - Executes a specific tool from the catalog
   ⚠️ REQUIRED: data.tool_name MUST be specified and exist in tool_catalog
   ```json
   {
     "id": "fetch_weather",
     "type": "tool",
     "label": "Get Weather Data",
     "data": {
       "tool_name": "weather_api",  // 🚨 REQUIRED for type="tool"
       "config": {"location": "London", "units": "celsius"},
       "ins": [],
       "outs": ["weather_data", "status"]
     }
   }
   ```

2. **agent** - Runs an AI agent with instructions and tools
   ⚠️ REQUIRED: data.agent_instructions MUST be specified
   ⚠️ OPTIONAL: data.tools - list of tool names the agent can use
   ```json
   {
     "id": "analyze_data",
     "type": "agent", 
     "label": "Data Analyst",
     "data": {
       "agent_instructions": "Analyze the weather data and provide insights",  // 🚨 REQUIRED for type="agent"
       "tools": ["calculator", "data_processor", "chart_generator"],  // 🔧 Tools the agent can use
       "config": {"model": "gpt-4", "temperature": 0.7},
       "ins": ["weather_data"],
       "outs": ["analysis", "insights"]
     }
   }
   ```
   
   🎯 **Agent-Tool Integration Principles**:
   - Agents autonomously decide when to use their tools during execution
   - Include all tools an agent might need for their task
   - Agent instructions should describe the goal, not tool usage details
   - Tools execute within agent reasoning, not as separate workflow steps

3. **decision** - Makes boolean or routing decisions (use decision tools or agents)
   ⚠️ REQUIRED: data.tool_name MUST be specified for decision tools
   ```json
   {
     "id": "check_rain",
     "type": "decision",
     "label": "Check Rain Condition", 
     "data": {
       "tool_name": "json_evaluator",  // 🚨 REQUIRED if using tool-based decision
       "config": {"expression": "data.weather.condition == 'rain'"},
       "ins": ["weather_data"],
       "outs": ["result", "details"]
     }
   }
   ```

4. **workflow_call** - Executes another workflow
   ⚠️ REQUIRED: data.workflow_id MUST be specified
   ```json
   {
     "id": "run_sub_workflow",
     "type": "workflow_call",
     "label": "Process Subset",
     "data": {
       "workflow_id": "data_processing_v2",  // 🚨 REQUIRED for type="workflow_call"
       "config": {"mode": "batch"},
       "ins": ["raw_data"],
       "outs": ["processed_data"]
     }
   }
   ```

📊 Data Flow Rules
------------------
1. **Port Naming**: Use clear, semantic names that match between nodes
   - Common Input ports: data, query, config, source, input, params, joke, text_input, user_input
   - Common Output ports: result, output, response, error, status, joke_output, analysis, summary
   - ⚠️ **CRITICAL**: Edge sourceHandle MUST match an actual output port in source node's "outs" array
   - ⚠️ **CRITICAL**: Edge targetHandle MUST match an actual input port in target node's "ins" array

2. **Edge Connections**: Connect compatible ports with matching names
   ```json
   // Source node has "outs": ["joke_output"]
   // Target node has "ins": ["joke_input"]
   {
     "id": "edge1",
     "source": "joke_creator",
     "target": "joke_evaluator", 
     "sourceHandle": "joke_output",    // Must exist in source node's outs
     "targetHandle": "joke_input"      // Must exist in target node's ins
   }
   ```

3. **Data References**: Use {node_id} or {node_id.field} syntax in config
   ```json
   "config": {
     "joke_to_evaluate": "{joke_creator}",           // Gets full result
     "temperature": "{weather_node.result.temp}"     // Gets specific field
   }
   ```

4. **Agent Node Data Flow**: For agent nodes, specify structured output format
   ```json
   {
     "id": "joke_creator",
     "type": "agent",
     "data": {
       "agent_instructions": "Create a funny joke. Output in JSON format: {\"joke_text\": \"your joke here\", \"category\": \"puns/wordplay/etc\"}",
       "ins": [],                    // No inputs needed
       "outs": ["joke_output"]       // Will output structured joke data
     }
   },
   {
     "id": "joke_evaluator", 
     "type": "agent",
     "data": {
       "agent_instructions": "Evaluate this joke: {joke_creator.joke_text}. Output in JSON: {\"rating\": 1-10, \"funny_reason\": \"explanation\"}",  // Reference specific field
       "ins": ["joke_input"],        // Expects joke input
       "outs": ["evaluation"]        // Will output structured evaluation
     }
   }
   ```
   
   🚨 **STRUCTURED AGENT OUTPUT REQUIREMENTS**:
   - Agents should output JSON when tools need to reference specific fields
   - Use format: "Output in JSON format: {\"field_name\": \"value\", \"number_field\": 123}"
   - Tools can then reference: `{agent_node.field_name}` or `{agent_node.number_field}`
   - For simple cases, plain text is fine: "Output the final answer as a number"

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

**🎯 TOOL PARAMETER CONFIGURATION EXAMPLES**

**🚨 CRITICAL DATA FLOW PRINCIPLE**: Tools should get their input data from OTHER nodes, not hardcoded values (unless it's initial configuration data like API endpoints, constants, etc.)

**🚨 TOOL USAGE PATTERNS**:
- **prompt_tool**: Use for INPUT generation or message passing at the BEGINNING/MIDDLE of workflows
- **user_input**: Use for collecting user input at the BEGINNING of workflows  
- **conditional_gate**: Use for routing/decision making in the MIDDLE of workflows
- **Agent nodes**: Can be FINAL output nodes - no tool needed after them for display
- **NEVER use prompt_tool as final output** - agents should be the final nodes that produce results

**✅ GOOD Examples:**

**Example 1: Riddle Solving with Tool-Enabled Agents**
```json
{
  "nodes": [
    {"id": "riddle_generator", "type": "agent", "data": {"agent_instructions": "Create a challenging arithmetic riddle", "ins": [], "outs": ["riddle"]}},
    {"id": "solver_agent", "type": "agent", "data": {"agent_instructions": "Solve the given arithmetic riddle using available math tools", "tools": ["add", "subtract", "multiply", "divide"], "ins": ["riddle"], "outs": ["solution"]}},
    {"id": "oracle_agent", "type": "agent", "data": {"agent_instructions": "Verify if the solver's solution is correct for the given riddle", "tools": ["add", "subtract", "multiply", "divide"], "ins": ["riddle", "solution"], "outs": ["verdict"]}}
  ],
  "edges": [
    {"source": "riddle_generator", "target": "solver_agent", "sourceHandle": "riddle", "targetHandle": "riddle"},
    {"source": "riddle_generator", "target": "oracle_agent", "sourceHandle": "riddle", "targetHandle": "riddle"},
    {"source": "solver_agent", "target": "oracle_agent", "sourceHandle": "solution", "targetHandle": "solution"}
  ]
}
```

**Example 2: Weather Analysis with Tool-Enabled Agent** 
```json
{
  "nodes": [
    {"id": "weather_analyst", "type": "agent", "data": {"agent_instructions": "Get weather for New York and Los Angeles, then compare and analyze the temperature difference", "tools": ["get_weather", "add", "subtract"], "ins": [], "outs": ["analysis"]}}
  ],
  "edges": []
}
```

**❌ BAD Examples - AVOID THESE:**
```json
// DON'T DO THIS - separate tool nodes for simple operations
{"id": "calculate", "type": "tool", "data": {"tool_name": "add", "config": {"a": 10, "b": 5}}} // Wrong!

// INSTEAD DO THIS - tool-enabled agents
{"id": "calculator_agent", "type": "agent", "data": {"agent_instructions": "Perform arithmetic calculations as needed", "tools": ["add", "subtract", "multiply", "divide"]}} // Correct!
```

**🚨 CRITICAL WORKFLOW RULES**:
1. **ALL REQUIRED PARAMETERS MUST BE PRESENT**: Every tool parameter from the catalog must be in the config
2. **🔥 DATA FLOW FIRST**: Tools should get data from OTHER nodes, NOT hardcoded values
   - ✅ CORRECT: `{"a": "{solver_agent.number}", "b": "{riddle_generator.value}"}` (gets data from other nodes)
   - ❌ WRONG: `{"a": 10, "b": 5}` (hardcoded values when data should come from workflow)
   - ✅ ACCEPTABLE: `{"api_key": "sk-12345", "endpoint": "https://api.com"}` (configuration constants)
3. **USE DATA FLOW REFERENCES**: Reference previous node outputs correctly based on tool type
   - **For user_input tools**: Use `{node_id}` directly (stores the input value)
     ✅ CORRECT: `{"a": "{user_input_node}", "multiplier": "{double_input_node}"}`
   - **For most other tools**: Use `{node_id.result}` for the main output
     ✅ CORRECT: `{"a": "{get_weather_ny.result}", "b": "{calculator_agent.result}"}`
   - **For complex outputs**: Use specific field access when tools return structured data
     ✅ CORRECT: `{"temp": "{weather_node.result.temperature}", "city": "{weather_node.result.city}"}`
   - ❌ WRONG: `{"a": "get_weather_ny.result"}` (missing braces)
   - ❌ WRONG: `{"a": "{user_input_node.input_value}"}` (user_input stores value directly)
4. **AGENT OUTPUTS**: When agents generate numbers or values, tools should reference those outputs:
   - If agent says "The answer is 42", tool should use `{"number": "{agent_node.result}"}`
   - If agent extracts values, tool should use `{"a": "{agent_node.first_number}", "b": "{agent_node.second_number}"}`
5. **NEVER LEAVE CONFIG EMPTY**: If a tool has parameters, config cannot be `{}`
6. **VALIDATE AGAINST CATALOG**: Check tool_catalog.parameters for required fields
7. **FIELD ACCESS FOR STRUCTURED DATA**: When tools return objects, access specific fields:
   - get_weather returns `{"temp": 70.5, "condition": "Clear"}`
   - To get temperature: `{get_weather_node.result.temp}`
   - To get condition: `{get_weather_node.result.condition}`

⚠️  **IMPORTANT**: This example uses tools like `add`, `multiply`, `divide` - verify these exist in your tool_catalog!

🔑 Key Rules:
- Decision nodes output structured data (true/false, route names)
- Router nodes consume decision data and output routing information
- Edges are NEVER conditional - they just carry data
- Use `conditional_router` or `boolean_mux` for path selection

🎯 Output Requirements
----------------------
Generate a WorkflowSpecLLM with:
- `title`: Clear, action-oriented name
- `description`: One sentence explaining the workflow purpose
- `nodes`: Array of nodes accomplishing the goal
- `edges`: Connections between nodes using node labels as source/target, with optional conditions
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
3. **🚨 PORT CONSISTENCY**: sourceHandle and targetHandle MUST match node ins/outs exactly
   - Edge sourceHandle must exist in source node's "outs" array
   - Edge targetHandle must exist in target node's "ins" array
4. **🚨 TOOL EXISTENCE**: All tool_name values MUST exist in the provided tool catalog
5. **No Orphaned Nodes**: Every node should be connected (except start/end nodes)
6. **Agent Data Flow**: For agent nodes, reference previous nodes' outputs in agent_instructions using {node_id} syntax

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
        self.last_workflow: Optional[WorkflowSpec] = None  # Track last generated workflow
        
        # Initialize the underlying agent with structured output
        self.agent = Agent(
            name="WorkflowPlanner",
            instructions=WORKFLOW_PLANNER_INSTRUCTIONS,
            model=model or os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            memory=memory,
            conversation_id=self.conversation_id,
            output_type=WorkflowSpecLLM,  # 🔑 guarantees structured JSON
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
        max_retries: int = 3,
        **kwargs
    ) -> tuple[WorkflowSpec, str]:
        """
        Generate a workflow specification from a user query with auto-retry on validation failures.
        
        Args:
            query: User's natural language description of the workflow
            tool_catalog: Available tools and their specifications
            context: Additional context for workflow generation
            max_retries: Maximum number of retries on validation failure (default: 3)
            **kwargs: Additional arguments passed to agent.run()
            
        Returns:
            WorkflowSpec: Validated workflow specification ready for execution
        """
        # Prepare context with tool catalog
        full_context = {
            "tool_catalog": tool_catalog or {},
            "additional_context": context or {}
        }
        
        # Get the JSON schema for WorkflowSpecLLM
        workflow_schema = WorkflowSpecLLM.model_json_schema()
        
        validation_errors = []
        attempt = 0
        
        while attempt <= max_retries:
            attempt += 1
            
            # Build the query with any previous validation errors
            error_feedback = ""
            if validation_errors:
                print(f"❌ Validation errors: {validation_errors}")
                error_feedback = f"""
❌ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERRORS:
{chr(10).join(f"- {error}" for error in validation_errors[-1])}

Please fix these specific issues in your next attempt:
{chr(10).join(f"{i+1}. {error}" for i, error in enumerate(validation_errors[-1]))}

"""
            
            # Add previous workflow context if available
            previous_workflow_context = ""
            if self.last_workflow:
                # Build comprehensive node descriptions
                node_descriptions = []
                for node in self.last_workflow.nodes:
                    node_desc = f"  - {node.id} ({node.type}): {node.label}"
                    if node.type == "tool":
                        node_desc += f" | tool: {node.data.tool_name}"
                    elif node.type == "agent":
                        # Show first 100 chars of instructions
                        inst_preview = node.data.agent_instructions[:100] + "..." if len(node.data.agent_instructions) > 100 else node.data.agent_instructions
                        node_desc += f" | instructions: {inst_preview}"
                        if node.data.tools:
                            node_desc += f" | tools: {node.data.tools}"
                    node_descriptions.append(node_desc)
                
                # Build edge descriptions
                edge_descriptions = []
                for edge in self.last_workflow.edges:
                    edge_desc = f"  - {edge.source} → {edge.target}"
                    if edge.data and edge.data.condition:
                        edge_desc += f" (condition: {edge.data.condition})"
                    edge_descriptions.append(edge_desc)
                
                previous_workflow_context = f"""
📝 PREVIOUS WORKFLOW (for reference):
Title: {self.last_workflow.title}
Description: {self.last_workflow.description}
Reasoning: {self.last_workflow.reasoning}

📊 NODES ({len(self.last_workflow.nodes)} total):
{chr(10).join(node_descriptions)}

🔗 EDGES ({len(self.last_workflow.edges)} total):
{chr(10).join(edge_descriptions) if edge_descriptions else "  - No edges defined"}

🔄 REFINEMENT MODE: You should preserve the overall structure and only modify what the user specifically requests. 
When user says "change X to Y", find node X and replace it with Y while keeping all connections intact.
"""

            # Format the query with context
            formatted_query = f"""
{error_feedback}{previous_workflow_context}
User Query: {query}

📋 EXPECTED OUTPUT SCHEMA:
{self._format_schema(workflow_schema)}

🔧 AVAILABLE TOOLS IN CATALOG:
{self._format_tool_catalog(tool_catalog or {})}

🔍 TOOL RETURN FORMATS:
- get_weather: Returns {{"temp": 70.5, "condition": "Clear"}}
  - Access temperature: {{node_id.result.temp}}
  - Access condition: {{node_id.result.condition}}
- add/calculator_add: Returns number (e.g., 135.5)
  - Access result: {{node_id.result}}
- subtract/multiply/divide: Returns number
  - Access result: {{node_id.result}}

🚨 CRITICAL REQUIREMENTS:
- You MUST ONLY use tools from the above catalog
- Use the EXACT tool names as listed
- 🎯 PREFER AGENT-TOOL INTEGRATION: For complex reasoning tasks, use agents with embedded tools
- Use type="tool" only for simple operations or external system integrations
- 🔧 AGENT TOOLS: When creating agent nodes, include relevant tools in the "tools" array
- ✅ DO USE: Agent with tools for reasoning + computation tasks
- 🚫 AVOID: Multiple separate tool nodes for related operations
- If required tools are missing, explain in the reasoning field
- DO NOT hallucinate or invent tool names
- Follow the exact schema structure above
- Ensure all nodes are connected (no orphaned nodes)
- Ensure all required fields are populated based on node type
- 🧠 AGENT DATA FLOW: Agents receive context from previous steps and reason with tools

Additional Context: {context or {}}

Generate a WorkflowSpecLLM that fulfills the user's requirements using ONLY the available tools above.
"""
            
            try:
                # Run the agent to generate the workflow with limited message history
                print(f"🔄 Workflow generation attempt {attempt}/{max_retries + 1}")
                result = await self.agent.run(
                    formatted_query,
                    conversation_id=self.conversation_id,
                    message_history_limit=5,  # Limit to last 5 messages to prevent context overflow
                    **kwargs
                )
                
                # Extract the structured output
                workflow_spec_llm = result.get("result")
                if not isinstance(workflow_spec_llm, WorkflowSpecLLM):
                    raise ValueError(f"Expected WorkflowSpecLLM, got {type(workflow_spec_llm)}")
                
                # Convert LLM spec to final spec with deterministic IDs
                workflow_spec = WorkflowSpec.from_llm_spec(workflow_spec_llm)
                
                # 🚨 CRITICAL VALIDATION: Check for tool hallucination and structural issues
                structural_issues = workflow_spec.validate_structure(tool_catalog or {})
                if structural_issues:
                    if attempt <= max_retries:
                        print(f"⚠️ Validation failed on attempt {attempt}, retrying with feedback...")
                        validation_errors.append(structural_issues)
                        continue
                    else:
                        error_msg = f"🚨 WORKFLOW VALIDATION FAILED AFTER {max_retries + 1} ATTEMPTS:\n" + "\n".join(structural_issues)
                        error_msg += f"\n\nWorkflow reasoning: {workflow_spec.reasoning}"
                        raise ValueError(error_msg)
                
                # Success! Store as last workflow for future context
                print(f"✅ Workflow validated successfully on attempt {attempt}")
                self.last_workflow = workflow_spec
                agent_response = result.get("conversation_id", "")
                return workflow_spec, agent_response
                
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                # Handle other errors
                if attempt <= max_retries:
                    print(f"❌ Error on attempt {attempt}: {str(e)}, retrying...")
                    validation_errors.append([f"Generation error: {str(e)}"])
                    continue
                else:
                    raise
        
        # Should not reach here
        raise ValueError(f"Failed to generate valid workflow after {max_retries + 1} attempts")
    
    def _format_tool_catalog(self, tool_catalog: dict) -> str:
        """Format the tool catalog for clear display to the LLM using rich parameter descriptions."""
        if not tool_catalog:
            return "❌ NO TOOLS AVAILABLE - Cannot create workflow without tools"
        
        formatted_tools = []
        for tool_name, tool_info in tool_catalog.items():
            # Format parameters with rich descriptions from pydantic-ai schema generation
            parameters = tool_info.get('parameters', {})
            required_params = tool_info.get('required_parameters', [])
            
            param_details = []
            for param_name, param_info in parameters.items():
                if isinstance(param_info, dict):
                    # New rich format with descriptions
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', 'No description')
                    is_required = param_info.get('required', param_name in required_params)
                    default_val = param_info.get('default')
                    
                    req_indicator = " (required)" if is_required else " (optional)"
                    default_info = f" [default: {default_val}]" if default_val is not None else ""
                    param_details.append(f"     • {param_name} ({param_type}){req_indicator}{default_info}: {param_desc}")
                else:
                    # Fallback for simple format
                    req_indicator = " (required)" if param_name in required_params else " (optional)"
                    param_details.append(f"     • {param_name} ({param_info}){req_indicator}")
            
            params_section = "\n".join(param_details) if param_details else "     • No parameters"
            
            # Add special notes for user_input tool
            usage_note = f'{{"tool_name": "{tool_name}", "config": {{ ... }} }}'
            if tool_name == 'user_input':
                usage_note += f'\n   🔗 Data Flow: Use `{{node_id}}` to reference user input (stores value directly)'
            
            formatted_tools.append(f"""
📦 {tool_name}
   Description: {tool_info.get('description', 'No description')}
   Parameters:
{params_section}
   Usage: {usage_note}
""")
        
        return f"""
Available Tools ({len(tool_catalog)} total):
{''.join(formatted_tools)}

🚨 REMINDER: Use ONLY these exact tool names. Any other tool will cause failure.
"""
    
    def _format_schema(self, schema: dict) -> str:
        """Format the JSON schema to highlight key requirements."""
        import json
        
        # Extract key parts we want to emphasize
        formatted = "WorkflowSpec Structure:\n"
        formatted += json.dumps(schema, indent=2)
        
        # Add specific callouts for required fields
        formatted += "\n\n🚨 FIELD REQUIREMENTS BY NODE TYPE:\n"
        formatted += "- For type='tool' nodes: data.tool_name is REQUIRED\n"
        formatted += "- For type='agent' nodes: data.agent_instructions is REQUIRED\n"
        formatted += "- For type='workflow_call' nodes: data.workflow_id is REQUIRED\n"
        formatted += "- For type='decision' nodes using tools: data.tool_name is REQUIRED\n"
        
        return formatted
    
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
        refined_spec: WorkflowSpec = result.get("result")
        # Patch: Convert WorkflowSpecLLM to WorkflowSpec if needed
        from ..data_models.workflow_spec import WorkflowSpecLLM, WorkflowSpec
        if isinstance(refined_spec, WorkflowSpecLLM):
            refined_spec = WorkflowSpec.from_llm_spec(refined_spec)
        if not isinstance(refined_spec, WorkflowSpec):
            raise ValueError(f"Expected WorkflowSpec after conversion, got {type(refined_spec)}, spec: {refined_spec}")
        # Store as last workflow for future context
        self.last_workflow = refined_spec
        return refined_spec
    
    def set_current_workflow(self, workflow_spec: WorkflowSpec):
        """
        Set the current workflow for context in future generations.
        
        Args:
            workflow_spec: WorkflowSpec to use as context for future generations
        """
        self.last_workflow = workflow_spec
    
    def get_current_workflow(self) -> Optional[WorkflowSpec]:
        """
        Get the current workflow being tracked for context.
        
        Returns:
            Current WorkflowSpec or None if no workflow is set
        """
        return self.last_workflow
    
    def clear_workflow_context(self):
        """Clear the current workflow context."""
        self.last_workflow = None
    
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
                    tool_name="web_fetch",  # Required for tool nodes
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
                    agent_instructions="Process the input data and extract key insights",  # Required for agent nodes
                    config={"model": "gpt-4o"},
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
                    tool_name="file_writer",  # Required for tool nodes
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