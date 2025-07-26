import os
import uuid
from typing import Dict, Any, Optional
from ...agents import Agent
from ...memory import AsyncMemory
from ..data_models.workflow_spec import WorkflowSpec, WorkflowSpecLLM, ROUTING_TOOLS
from datetime import datetime
from ...utilities.io_logger import log_prompt
from .workflow_prompts import get_workflow_planner_instructions


class WorkflowPromptBuilder:
    """
    Encapsulates all prompt construction logic for the WorkflowPlanner.
    Provides clean separation between prompt building and execution logic.
    """
    
    def __init__(self, query: str, tool_catalog: Dict[str, Any], schema: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self.query = query
        self.tool_catalog = tool_catalog or {}
        self.schema = schema
        self.context = context or {}
        self.validation_errors: list[list[str]] = []
        self.previous_workflow: Optional['WorkflowSpec'] = None
        
    def set_validation_errors(self, errors: list[list[str]]) -> 'WorkflowPromptBuilder':
        """Set validation errors from previous attempts."""
        self.validation_errors = errors
        return self
        
    def set_previous_workflow(self, workflow: Optional['WorkflowSpec']) -> 'WorkflowPromptBuilder':
        """Set previous workflow for refinement context."""
        self.previous_workflow = workflow
        return self
    
    def build(self) -> str:
        """Build the complete prompt for the workflow planner."""
        # Split tools and data sources
        tools_section, data_sources_section = self._split_tools_and_data_sources(self.tool_catalog)
        
        parts = [
            self._build_error_feedback(),
            self._build_previous_workflow_context(), 
            f"User Query: {self.query}",
            "",
            "📋 EXPECTED OUTPUT SCHEMA:",
            self._format_schema(self.schema),
            "",
            tools_section,
            "",
            data_sources_section,
            "",
            self._build_tool_return_formats(),
            "",
            self._build_critical_requirements(),
            "",
            f"Additional Context: {self.context}",
            "",
            "Generate a WorkflowSpecLLM that fulfills the user's requirements using ONLY the available tools and data sources above."
        ]
        
        return "\n".join(part for part in parts if part is not None)
    
    def _split_tools_and_data_sources(self, tool_catalog: Dict[str, Any]) -> tuple[str, str]:
        """Split into tools (for agents) and data sources (completely separate)."""
        from ..data_models.data_source_registry import get_valid_data_source_names, get_data_source_description
        
        # Tools are just the tool_catalog as-is (no data sources in there)
        tools = tool_catalog
        
        # Data sources are completely separate - build from registry
        data_sources = {}
        for source_name in get_valid_data_source_names():
            if source_name == "user_input":
                data_sources[source_name] = {
                    "name": source_name,
                    "description": get_data_source_description(source_name),
                    "parameters": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to show to the user when collecting input",
                            "required": True
                        }
                    },
                    "required_parameters": ["prompt"]
                }
            elif source_name == "prompt_tool":
                data_sources[source_name] = {
                    "name": source_name,
                    "description": get_data_source_description(source_name),
                    "parameters": {
                        "message": {
                            "type": "string",
                            "description": "The prompt/context message to inject into the workflow",
                            "required": True
                        }
                    },
                    "required_parameters": ["message"]
                }
        
        # Format sections
        tools_section = self._format_catalog_section("🔧 AVAILABLE TOOLS (for agent nodes)", tools, "Use these in agent/decision nodes' tools array")
        data_sources_section = self._format_catalog_section("📋 AVAILABLE DATA SOURCES (for data_source nodes)", data_sources, "Use these as source_name in data_source nodes")
        
        return tools_section, data_sources_section
    
    def _format_catalog_section(self, title: str, catalog: Dict[str, Any], usage_note: str) -> str:
        """Format a section of the catalog (tools or data sources)."""
        if not catalog:
            return f"{title}:\n❌ NO ITEMS AVAILABLE"
        
        formatted_items = []
        for item_name, item_info in catalog.items():
            # Format parameters with rich descriptions
            parameters = item_info.get('parameters', {})
            required_params = item_info.get('required_parameters', [])
            
            param_details = []
            for param_name, param_info in parameters.items():
                if isinstance(param_info, dict):
                    # Rich format with descriptions
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
            
            # Usage example
            usage_example = f'{{"source_name": "{item_name}", "config": {{ ... }} }}' if "DATA SOURCES" in title else f'"{item_name}"'
            if item_name == 'user_input':
                usage_example += '\n   🔗 Data Flow: Use `{node_id}` to reference user input (stores value directly)'
            
            formatted_items.append(f"""
📦 {item_name}
   Description: {item_info.get('description', 'No description')}
   Parameters:
{params_section}
   Usage: {usage_example}
""")
        
        return f"""{title} ({len(catalog)} total):
{''.join(formatted_items)}

🚨 {usage_note}. Any other names will cause failure."""
    
    def _build_error_feedback(self) -> Optional[str]:
        """Build validation error feedback section."""
        if not self.validation_errors:
            return None
            
        latest_errors = self.validation_errors[-1]
        
        # Provide specific guidance based on error types
        guidance = []
        for error in latest_errors:
            if "MISSING ROUTE_INDEX" in error:
                guidance.append("🔧 FIX: Add route_index to edges from decision nodes. Example: edge.data.route_index = 0 for first route, 1 for second route")
            elif "DANGLING ROUTING NODE" in error:
                guidance.append("🔧 FIX: Decision nodes MUST have outgoing edges to route to different agents based on conditions")
            elif "SOURCE HALLUCINATION" in error:
                guidance.append("🔧 FIX: Use only valid tool names from the provided tool catalog. Check spelling and availability")
            elif "ORPHANED ROUTE_INDEX" in error:
                guidance.append("🔧 FIX: Only add route_index to edges FROM decision nodes that have conditional_gate tool")
            elif "SLA MISCONFIGURATION" in error:
                guidance.append("🔧 FIX: Ensure SLA requirements match available tools. The conditional_gate tool must be in the tools list")
            else:
                guidance.append(f"🔧 FIX: {error}")
        
        return f"""❌ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERRORS:
{chr(10).join(f"- {error}" for error in latest_errors)}

🛠️ SPECIFIC FIXES REQUIRED:
{chr(10).join(guidance)}

📋 REMINDER OF KEY RULES:
1. Decision nodes MUST have the conditional_gate tool
2. Decision nodes need outgoing edges with route_index (0, 1, etc.) for routing
3. Only use tools that exist in the provided tool catalog
4. Nodes after decision gates should use execution_mode: "for_each"

"""
    
    def _build_previous_workflow_context(self) -> Optional[str]:
        """Build previous workflow context for refinement."""
        if not self.previous_workflow:
            return None
            
        # Build comprehensive node descriptions
        node_descriptions = []
        for node in self.previous_workflow.nodes:
            node_desc = f"  - {node.id} ({node.type}): {node.label}"
            if node.type == "data_source":
                node_desc += f" | source: {node.data.source_name}"
            elif node.type == "agent":
                # Show first 100 chars of instructions
                if node.data.agent_instructions:
                    inst_preview = node.data.agent_instructions[:100] + "..." if len(node.data.agent_instructions) > 100 else node.data.agent_instructions
                    node_desc += f" | instructions: {inst_preview}"
                if node.data.tools:
                    node_desc += f" | tools: {node.data.tools}"
            node_descriptions.append(node_desc)
        
        # Build edge descriptions
        edge_descriptions = []
        for edge in self.previous_workflow.edges:
            edge_desc = f"  - {edge.source} → {edge.target}"
            if edge.data and edge.data.condition:
                edge_desc += f" (condition: {edge.data.condition})"
            edge_descriptions.append(edge_desc)
        
        return f"""📝 PREVIOUS WORKFLOW (for reference):
Title: {self.previous_workflow.title}
Description: {self.previous_workflow.description}
Reasoning: {self.previous_workflow.reasoning}

📊 NODES ({len(self.previous_workflow.nodes)} total):
{chr(10).join(node_descriptions)}

🔗 EDGES ({len(self.previous_workflow.edges)} total):
{chr(10).join(edge_descriptions) if edge_descriptions else "  - No edges defined"}

🔄 REFINEMENT MODE: You should preserve the overall structure and only modify what the user specifically requests. 
When user says "change X to Y", find node X and replace it with Y while keeping all connections intact.

"""
    
    def _build_tool_return_formats(self) -> str:
        """Build tool return format documentation."""
        return """🔍 TOOL RETURN FORMATS:
- get_weather: Returns {"temp": 70.5, "condition": "Clear"}
  - Access temperature: {node_id.result.temp}
  - Access condition: {node_id.result.condition}
- add/calculator_add: Returns number (e.g., 135.5)
  - Access result: {node_id.result}
- subtract/multiply/divide: Returns number
  - Access result: {node_id.result}"""
    
    def _build_critical_requirements(self) -> str:
        """Build critical requirements section."""
        return """🚨 CRITICAL REQUIREMENTS:
- You MUST ONLY use tools from the above catalog
- Use the EXACT tool names as listed
- 🎯 PREFER AGENT-TOOL INTEGRATION: For complex reasoning tasks, use agents with embedded tools
- 🔧 AGENT TOOLS: When creating agent nodes, include relevant tools in the "tools" array
- ✅ DO USE: Agent with tools for reasoning + computation tasks
- 🚫 AVOID: Multiple separate tool nodes for related operations
- If required tools are missing, explain in the reasoning field
- DO NOT hallucinate or invent tool names
- Follow the exact schema structure above
- Ensure all nodes are connected (no orphaned nodes)
- Ensure all required fields are populated based on node type
- 🧠 AGENT DATA FLOW: Agents receive context from previous steps and reason with tools"""
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the JSON schema to highlight key requirements."""
        import json
        
        # Extract key parts we want to emphasize
        formatted = "WorkflowSpec Structure:\n"
        formatted += json.dumps(schema, indent=2)
        
        # Add specific callouts for required fields
        formatted += "\n\n🚨 FIELD REQUIREMENTS BY NODE TYPE:\n"
        formatted += "- For type='data_source' nodes: data.source_name is REQUIRED\n"
        formatted += "- For agent types (agent, decision): data.agent_instructions is REQUIRED\n"
        formatted += "- For type='workflow_call' nodes: data.workflow_id is REQUIRED\n"
        formatted += "- For decision nodes: Use agent_instructions + optional tools array (NOT source_name)\n"
        
        return formatted
    


def log_full_prompt_and_response(prompt: str, response: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, attempt: Optional[int] = None):
    """Centralized logging for workflow planner prompts and responses."""
    # Add attempt info to metadata
    if metadata is None:
        metadata = {}
    if attempt is not None:
        metadata["attempt"] = attempt
    
    # Log the full prompt
    log_prompt(
        prompt_type="workflow_planner_full",
        prompt=prompt,
        metadata=metadata
    )
    
    # Log the response if provided
    if response is not None:
        log_prompt(
            prompt_type="workflow_planner_response", 
            prompt=prompt,  # Reference to original prompt
            response=str(response),
            metadata=metadata
        )


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
            instructions=get_workflow_planner_instructions(),  # Dynamic prompt with valid data sources
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
        message_history_limit: int = 5,
        max_retries: int = 3,
        **kwargs
    ) -> WorkflowSpec:
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
        
        # Get the JSON schema for WorkflowSpecLLM
        workflow_schema = WorkflowSpecLLM.model_json_schema()
        
        validation_errors = []
        attempt = 0
        
        # Create the prompt builder
        prompt_builder = WorkflowPromptBuilder(
            query=query,
            tool_catalog=tool_catalog or {},
            schema=workflow_schema,
            context=context
        ).set_previous_workflow(self.last_workflow)

        uuid_for_workflow = uuid.uuid4()
        
        while attempt <= max_retries:
            attempt += 1
            
            try:
                print(f"🔄 Workflow generation attempt {attempt}/{max_retries + 1}")
                
                # Build context information containing schema, tools, etc.
                context_info = prompt_builder.set_validation_errors(validation_errors).build()
                
                # Set the context on the agent before running
                self.agent.set_context(context_info)
                
                # Filter out conversation_id from kwargs to prevent duplicate parameter error
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'conversation_id'}
                
                # Centralized logging
                log_full_prompt_and_response(
                    prompt=f"Query: {query}\n\n==Context==\n{context_info}",  # Log full prompt for debugging
                    metadata={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "conversation_id": self.conversation_id,
                        "tool_catalog_size": len(tool_catalog or {}),
                        "has_validation_feedback": bool(validation_errors),
                        "query_length": len(query),
                        "context_info_length": len(context_info),
                        "instructions_length": len(get_workflow_planner_instructions())
                    },
                    attempt=attempt
                )
                
                # Execute the LLM call with just the user's query
                result = await self.agent.run(
                    query,  # Just pass the user's simple request like "stock agent"
                    conversation_id=self.conversation_id,
                    message_history_limit=message_history_limit,  # Limit to prevent context overflow
                    **filtered_kwargs
                )
                
                # Log the response with full context information
                log_full_prompt_and_response(
                    prompt=f"Query: {query}",  # Log full prompt for debugging
                    response=str(result),
                    metadata={
                        "attempt": attempt,
                        "conversation_id": self.conversation_id,
                        "result_type": type(result.get("result")).__name__ if result else "None",
                        "response_length": len(str(result)) if result else 0,
                        "query": query,
                        "context_length": len(context_info)
                    },
                    attempt=attempt
                )
                
                # Extract the structured output
                workflow_spec_llm = result.get("result")
                if not isinstance(workflow_spec_llm, WorkflowSpecLLM):
                    raise ValueError(f"Expected WorkflowSpecLLM, got {type(workflow_spec_llm)}")
                
                # Use standardized workflow generation reporting for mission-critical debugging
                if workflow_spec_llm.nodes:
                    # Convert to WorkflowSpec for standardized reporting using single source of truth
                    temp_workflow = WorkflowSpec.from_llm_spec(workflow_spec_llm, uuid_for_workflow, rev=attempt)
                    
                    # Generate standardized workflow generation report
                    generation_report = self._create_workflow_generation_report(
                        query=query, 
                        attempt=attempt, 
                        max_retries=max_retries,
                        workflow_spec=temp_workflow
                    )
                    
                    # Log using our standardized reporting system
                    from ...utilities.io_logger import system_logger
                    system_logger.info(
                        "WorkflowPlanner generation analysis", 
                        data={"generation_report": generation_report}
                    )
                    
                    # Also print the full report to console for visibility
                    print("\n" + "="*70)
                    print("📊 FULL WORKFLOW GENERATION REPORT")
                    print("="*70)
                    print(generation_report)
                    print("="*70 + "\n")
                
                # Check if this is a chat-only response (nodes/edges are null)
                if workflow_spec_llm.nodes is None or workflow_spec_llm.edges is None:
                    print(f"💬 Chat-only response detected: nodes={workflow_spec_llm.nodes}, edges={workflow_spec_llm.edges}")
                    # Return the WorkflowSpecLLM directly - no conversion needed
                    return workflow_spec_llm
                
                # Convert LLM spec to final spec with deterministic IDs
                workflow_spec = WorkflowSpec.from_llm_spec(workflow_spec_llm, uuid_for_workflow, rev=attempt)
                
                # 🚨 CRITICAL VALIDATION: Check for tool hallucination and structural issues
                # Build combined catalog for validation (tools + data sources)
                from ..data_models.data_source_registry import get_valid_data_source_names, get_data_source_description
                
                validation_catalog = (tool_catalog or {}).copy()
                
                # Add data sources for validation
                for source_name in get_valid_data_source_names():
                    if source_name == "user_input":
                        validation_catalog[source_name] = {
                            "name": source_name,
                            "description": get_data_source_description(source_name),
                            "parameters": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to show to the user when collecting input",
                                    "required": True
                                }
                            },
                            "required_parameters": ["prompt"]
                        }
                    elif source_name == "prompt_tool":
                        validation_catalog[source_name] = {
                            "name": source_name,
                            "description": get_data_source_description(source_name),
                            "parameters": {
                                "message": {
                                    "type": "string",
                                    "description": "The prompt/context message to inject into the workflow",
                                    "required": True
                                }
                            },
                            "required_parameters": ["message"]
                        }
                
                structural_issues = workflow_spec.validate_structure(validation_catalog)
                if structural_issues:
                    if attempt <= max_retries:
                        print(f"⚠️ Validation failed on attempt {attempt}, retrying with feedback...")
                        print("📋 VALIDATION FEEDBACK REPORT:")
                        print("─" * 60)
                        for i, issue in enumerate(structural_issues, 1):
                            print(f"  {i}. {issue}")
                        print("─" * 60)
                        validation_errors.append(structural_issues)
                        continue
                    else:
                        error_msg = f"🚨 WORKFLOW VALIDATION FAILED AFTER {max_retries + 1} ATTEMPTS:\n" + "\n".join(structural_issues)
                        error_msg += f"\n\nWorkflow reasoning: {workflow_spec.reasoning}"
                        raise ValueError(error_msg)
                
                # Success! Store as last workflow for future context
                print(f"✅ Workflow validated successfully on attempt {attempt}")
                self.last_workflow = workflow_spec
                return workflow_spec
                
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
    
    # _format_tool_catalog and _format_schema methods moved to WorkflowPromptBuilder class
    
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
{workflow_spec.to_llm_prompt()}

USER FEEDBACK: {feedback}

Please generate an improved WorkflowSpec that addresses the feedback while maintaining the core functionality.
Reference the topology and SLA requirements above when making changes.
"""
        result = await self.agent.run(
            refinement_query,
            conversation_id=self.conversation_id,
            message_history_limit=7,  # Limit to last 7 messages to prevent context overflow
            **kwargs
        )
        refined_spec_llm = result.get("result")
        
        # Check if this is a chat-only response (nodes/edges are null)
        if isinstance(refined_spec_llm, WorkflowSpecLLM):
            if refined_spec_llm.nodes is None or refined_spec_llm.edges is None:
                # Return the WorkflowSpecLLM directly for chat-only responses
                return refined_spec_llm
        
        # Convert WorkflowSpecLLM to WorkflowSpec if needed
        if isinstance(refined_spec_llm, WorkflowSpecLLM):
            refined_spec = WorkflowSpec.from_llm_spec(refined_spec_llm)
        else:  #huh?
            refined_spec = refined_spec_llm
            
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
    
    def _create_workflow_generation_report(self, query: str, attempt: int, max_retries: int, workflow_spec: WorkflowSpec) -> str:
        """Generate standardized workflow generation analysis report using single source of truth."""
        #TODO: add Workflow SLA: {workflow_spec.sla}
        # Header with generation context
        header = f"""🤖 WORKFLOW GENERATION ANALYSIS
================================
User Query: "{query}"
Generation Attempt: {attempt}/{max_retries + 1}
Workflow Title: {workflow_spec.title}
Reasoning: {workflow_spec.reasoning}
Generated Nodes: {len(workflow_spec.nodes)}
Generated Edges: {len(workflow_spec.edges)}
Generation Timestamp: {datetime.now().isoformat()}

"""
        
        # Analyze node type distribution and flag potential issues
        node_analysis = []
        tool_nodes = []
        agent_nodes = []
        
        for node in workflow_spec.nodes:
            if node.type == 'data_source':
                tool_name = getattr(node.data, 'source_name', 'unknown')
                tool_nodes.append(f"  🔧 {node.label} (source: {tool_name})")
                
            else:
                agent_tools = getattr(node.data, 'tools', [])
                agent_nodes.append(f"  🤖 {node.label} ({node.type}) with tools: {agent_tools}")
        
        if tool_nodes:
            node_analysis.extend([
                "🚨 TOOL NODES DETECTED:",
                *tool_nodes,
                ""
            ])
        
        if agent_nodes:
            node_analysis.extend([
                "✅ AGENT NODES CREATED:",
                *agent_nodes,
                ""
            ])
        
        # Use our single source of truth for complete workflow representation
        workflow_representation = f"""
📋 COMPLETE PREVIOUS WORKFLOW SPECIFICATION (to correct if it has issues):
{workflow_spec.to_llm_prompt()}
"""
        
        return header + '\n'.join(node_analysis) + workflow_representation
    
    # def create_example_workflow(self, title: str = "Example Workflow") -> WorkflowSpec:
    #     """
    #     Create a simple example workflow for testing.
        
    #     Args:
    #         title: Title for the example workflow
            
    #     Returns:
    #         WorkflowSpec: Example workflow specification
    #     """
    #     from ..data_models.workflow_spec import NodeSpec, NodeData, EdgeSpec, EdgeData
        
    #     workflow_id = uuid.uuid4()
        
    #     # Create example nodes
    #     nodes = [
    #         NodeSpec(
    #             id="start",
    #             type="tool",
    #             label="Fetch Data",
    #             data=NodeData(
    #                 tool_name="web_fetch",  # Required for tool nodes
    #                 config={"source": "api", "endpoint": "/data"},
    #                 ins=["trigger"],
    #                 outs=["data", "status"]
    #             ),
    #             runtime={"timeout": 30}
    #         ),
    #         NodeSpec(
    #             id="process",
    #             type="agent",
    #             label="Process Data",
    #             data=NodeData(
    #                 agent_instructions="Process the input data and extract key insights",  # Required for agent nodes
    #                 config={"model": "gpt-4o"},
    #                 ins=["data"],
    #                 outs=["result", "summary"]
    #             ),
    #             runtime={"timeout": 60}
    #         ),
    #         NodeSpec(
    #             id="output",
    #             type="tool",
    #             label="Save Result",
    #             data=NodeData(
    #                 tool_name="file_writer",  # Required for tool nodes
    #                 config={"destination": "file", "format": "json"},
    #                 ins=["result"],
    #                 outs=["success"]
    #             ),
    #             runtime={"timeout": 15}
    #         )
    #     ]
        
    #     # Create edges
    #     edges = [
    #         EdgeSpec(
    #             id="start_to_process",
    #             source="start",
    #             target="process",
    #             sourceHandle="data",
    #             targetHandle="data",
    #             data=EdgeData(condition="status == 'success'")
    #         ),
    #         EdgeSpec(
    #             id="process_to_output",
    #             source="process",
    #             target="output",
    #             sourceHandle="result",
    #             targetHandle="result"
    #         )
    #     ]
        
    #     return WorkflowSpec(
    #         id=workflow_id,
    #         rev=1,
    #         title=title,
    #         description="A simple example workflow that fetches, processes, and saves data",
    #         nodes=nodes,
    #         edges=edges,
    #         metadata={
    #             "created_at": "2024-01-01T00:00:00Z",
    #             "tags": ["example", "demo"]
    #         }
    #     )