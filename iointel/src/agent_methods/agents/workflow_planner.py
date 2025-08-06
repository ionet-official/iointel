import os
import uuid
from typing import Dict, Any, Optional
from iointel.src.agents import Agent
from iointel.src.memory import AsyncMemory
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, WorkflowSpecLLM, ROUTING_TOOLS
from datetime import datetime
from iointel.src.utilities.io_logger import log_prompt, get_component_logger
from iointel.src.utilities.unified_prompt_system import unified_prompt_system, PromptType
from iointel.src.utilities.conversion_utils import (
    validation_errors_to_llm_prompt,
    workflow_spec_to_llm_prompt,
    tool_catalog_to_llm_prompt
)
from iointel.src.agent_methods.agents.workflow_prompts import get_workflow_planner_instructions


class WorkflowPromptBuilder:
    """
    Encapsulates all prompt construction logic for the WorkflowPlanner.
    Provides clean separation between prompt building and execution logic.
    
    TODO: Migrate to unified_prompt_system for better reusability and search.
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
        """Build the complete prompt using unified prompt system."""
        # Get workflow planner instructions from unified system
        templates = unified_prompt_system.search_templates(
            prompt_type=PromptType.AGENT_INSTRUCTIONS,
            tags=["workflow_planner"]
        )
        
        if not templates:
            # Fallback to basic prompt if template not found
            return f"Generate a workflow for: {self.query}\nUse these tools: {self.tool_catalog}"
        
        # Get data sources for template variables
        from iointel.src.agent_methods.data_models.data_source_registry import (
            get_valid_data_source_names, 
            create_data_source_knowledge_section
        )
        
        valid_sources = get_valid_data_source_names()
        sources_list = "', '".join(valid_sources)
        data_source_knowledge = create_data_source_knowledge_section()
        
        # Render the template with dynamic data
        prompt_instance = unified_prompt_system.render_prompt(
            templates[0].id,
            VALID_DATA_SOURCES=f"'{sources_list}'",
            DATA_SOURCE_KNOWLEDGE=data_source_knowledge
        )
        
        # Build additional context sections - PUT CRITICAL INSTRUCTIONS FIRST!
        parts = [
            "🚨🚨🚨 CRITICAL: READ THIS FIRST - DATA SOURCE CONFIG IS MANDATORY 🚨🚨🚨",
            "Every data_source node MUST have config with ALL required parameters.",
            "",
            "COPY THESE EXACT TEMPLATES:",
            "user_input: {\"type\": \"data_source\", \"label\": \"Your Label\", \"data\": {\"source_name\": \"user_input\", \"config\": {\"message\": \"Your message\", \"default_value\": \"Your default\"}}}",
            "prompt_tool: {\"type\": \"data_source\", \"label\": \"Your Label\", \"data\": {\"source_name\": \"prompt_tool\", \"config\": {\"message\": \"Your message\", \"default_value\": \"Your default\"}}}",
            "",
            "🚨 NEVER use config: null or config: {} - ALWAYS include message and default_value",
            "=" * 80,
            "",
            self._build_error_feedback(),
            self._build_previous_workflow_context(), 
            self._build_refinement_guidance() if self.previous_workflow else None,
            prompt_instance.content,  # Use the unified prompt system content
            "",
            f"User Query: {self.query}",
            "",
            "📋 EXPECTED OUTPUT SCHEMA:",
            self._format_schema(self.schema),
            "",
            self._build_tools_section(),
            "",
            f"Additional Context: {self.context}",
            "",
            "Generate a WorkflowSpecLLM that fulfills the user's requirements using ONLY the available tools and data sources above."
        ]
        
        return "\n".join(part for part in parts if part is not None)
    
    def _build_error_feedback(self) -> str:
        """Build validation error feedback using centralized conversion utils."""
        if not self.validation_errors:
            return ""
        
        from iointel.src.utilities.conversion_utils import validation_errors_to_llm_prompt
        return validation_errors_to_llm_prompt(self.validation_errors)
    
    def _build_refinement_guidance(self) -> str:
        """Build refinement guidance when a previous workflow exists."""
        if not self.previous_workflow:
            return ""
        
        return f"""
🔄 WORKFLOW REFINEMENT REQUEST:
The user is asking to refine or improve an existing workflow. 
Consider the previous workflow structure and user feedback to make targeted improvements.

Previous workflow: {self.previous_workflow.title}
"""
    
    def _build_tools_section(self) -> str:
        """Build tools section using existing logic."""
        tools_section, data_sources_section = self._split_tools_and_data_sources(self.tool_catalog)
        return f"{tools_section}\n\n{data_sources_section}"
    
    def _split_tools_and_data_sources(self, tool_catalog: Dict[str, Any]) -> tuple[str, str]:
        """Split into tools (for agents) and data sources (completely separate)."""
        from ...utilities.tool_registry_utils import create_data_source_catalog
        
        # Data sources are completely separate - build from centralized registry
        data_sources = create_data_source_catalog()
        
        # Format sections
        tools_section = self._format_catalog_section("🔧 AVAILABLE TOOLS (for agent nodes)", tool_catalog, "Use these in agent/decision nodes' tools array")
        data_sources_section = self._format_catalog_section("📋 AVAILABLE DATA SOURCES (for data_source nodes)", data_sources, "Use these as source_name in data_source nodes")
        
        return tools_section, data_sources_section
    
    def _format_catalog_section(self, title: str, catalog: Dict[str, Any], usage_note: str) -> str:
        """Format a section of the catalog using centralized converter."""
        if not catalog:
            return f"{title}:\n❌ NO ITEMS AVAILABLE"
        
        # Use centralized converter for tool catalog
        catalog_prompt = tool_catalog_to_llm_prompt(catalog)
        
        # Replace the generic header with our specific title
        catalog_prompt = catalog_prompt.replace("# Available Tools:", title)
        
        # Add usage note and data source examples if needed
        suffix = f"\n🚨 {usage_note}. Any other names will cause failure."
        
        if "DATA SOURCES" in title:
            # Add mandatory config examples for data sources - CLEAR AS FUCK
            suffix = f"""

🚨🚨🚨 CRITICAL: DATA SOURCE CONFIG IS MANDATORY 🚨🚨🚨
Every data_source node MUST have config with ALL required parameters.

COPY THESE EXACT TEMPLATES (filled in with appropriate values for the users query):

user_input template:
{{
  "type": "data_source",
  "label": "Your Label Here", 
  "data": {{
    "source_name": "user_input",
    "config": {{
      "message": "Your specific message text here",
      "default_value": "Your default value here"
    }}
  }}
}}

prompt_tool template:
{{
  "type": "data_source",
  "label": "Your Label Here",
  "data": {{
    "source_name": "prompt_tool", 
    "config": {{
      "message": "Your specific message text here",
      "default_value": "Your default value here"
    }}
  }}
}}

🚨 NEVER use config: null or config: {{}} - ALWAYS include message and default_value
{suffix}"""
        
        return f"{catalog_prompt}{suffix}"

    
    def _build_previous_workflow_context(self) -> Optional[str]:
        """Build previous workflow context for refinement using centralized converter."""
        if not self.previous_workflow:
            return None
        
        # Use centralized converter to get workflow representation
        workflow_prompt = workflow_spec_to_llm_prompt(self.previous_workflow)
        
        # Add refinement-specific guidance
        refinement_guidance = f"""
📝 PREVIOUS WORKFLOW (for reference):
{workflow_prompt}

🔄 REFINEMENT MODE: You should preserve the overall structure and only modify what the user specifically requests. 
When user says "change X to Y", find node X and replace it with Y while keeping all connections intact.
When user says "remove X" or "Change tool X to required as last step", remove only that specific item (tool, node, etc.) and keep everything else.
"""
        
        return refinement_guidance
    
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
- 🧠 AGENT DATA FLOW: Agents receive context from previous steps and reason with tools

🚨 DATA SOURCE CONFIG REQUIREMENTS (CRITICAL):
- Data source config requirements are shown in the DATA SOURCES section above
- NEVER use empty config - all parameters are mandatory
- Follow the exact templates provided in the DATA SOURCES section"""
    
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
    
    logger = get_component_logger("WORKFLOW_PLANNER", grouped=True)
    
    def __init__(
        self,
        memory: Optional[AsyncMemory] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        debug: bool = False,
        validation_bypass: bool = False,
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
        self.validation_bypass = validation_bypass  # For debugging broken validation
        
        if self.validation_bypass:
            print("⚠️ WorkflowPlanner initialized with validation bypass enabled")
        
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
        
        with self.logger.group(f"Workflow Generation: {query[:100]}{'...' if len(query) > 100 else ''}"):
            while attempt <= max_retries:
                attempt += 1
                
                with self.logger.group(f"Attempt {attempt}/{max_retries + 1}"):
                    try:
                        self.logger.info(f"Starting generation attempt {attempt}")
                        
                        # Build context information containing schema, tools, etc.
                        context_info = prompt_builder.set_validation_errors(validation_errors).build()
                        
                        # Log context info summary
                        self.logger.info("Context prepared", data={
                            "query_length": len(query),
                            "tool_count": len(tool_catalog or {}),
                            "has_validation_feedback": bool(validation_errors),
                            "context_length": len(context_info)
                        })
                        
                        # Log validation errors if any
                        if validation_errors:
                            with self.logger.group("Previous Validation Errors"):
                                for attempt_idx, errors in enumerate(validation_errors, 1):
                                    self.logger.warning(f"Attempt {attempt_idx} errors:", data={
                                        "error_count": len(errors),
                                        "errors": errors
                                    })
                        
                        # Set the context on the agent before running
                        self.agent.set_context(context_info)
                        
                        # Filter out conversation_id from kwargs to prevent duplicate parameter error
                        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'conversation_id'}
                        
                        # Log the full prompt context
                        with self.logger.group("Prompt Context"):
                            # Log summary of what's in the prompt
                            self.logger.info("Full prompt context built", data={
                                "context_length": len(context_info),
                                "has_critical_datasource_warning": "🚨🚨🚨 CRITICAL: READ THIS FIRST" in context_info,
                                "has_error_feedback": bool(validation_errors),
                                "has_previous_workflow": prompt_builder.previous_workflow is not None,
                                "previous_workflow_title": prompt_builder.previous_workflow.title if prompt_builder.previous_workflow else None
                            })
                            
                            # If we have validation errors, show what specific feedback we're giving
                            if validation_errors:
                                feedback_summary = []
                                for errors in validation_errors:
                                    for error in errors:
                                        if "MISSING PARAMETERS" in error:
                                            feedback_summary.append("Missing required parameters in data source")
                                        elif "EMPTY CONFIG" in error:
                                            feedback_summary.append("Empty config in data source")
                                        elif "HALLUCINATION" in error:
                                            feedback_summary.append("Using non-existent tools/sources")
                                
                                self.logger.info("Validation feedback being provided", data={
                                    "feedback_types": list(set(feedback_summary))
                                })
                        
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
                        
                        # Log the full prompt to UI for transaction visibility
                        from ...utilities.io_logger import log_prompt
                        
                        full_prompt = f"CONTEXT:\n{context_info}\n\nUSER QUERY:\n{query}"
                        log_prompt(
                            prompt_type=f"workflow_generation_attempt_{attempt}",
                            prompt=full_prompt,
                            metadata={
                                "attempt": attempt,
                                "max_retries": max_retries,
                                "query": query,
                                "conversation_id": self.conversation_id,
                                "tool_count": len(tool_catalog) if tool_catalog else 0,
                                "context_size": len(context_info)
                            }
                        )
                        
                        # Execute the LLM call with just the user's query
                        with self.logger.group("LLM Generation"):
                            self.logger.info("Calling LLM with query", data={"query": query})
                            
                            result = await self.agent.run(
                                query,  # Just pass the user's simple request like "stock agent"
                                conversation_id=self.conversation_id,
                                message_history_limit=message_history_limit,  # Limit to prevent context overflow
                                **filtered_kwargs
                            )
                            
                            self.logger.info("LLM responded", data={
                                "result_type": type(result.get("result")).__name__ if result else "None",
                                "has_nodes": bool(result.get("result") and getattr(result.get("result"), "nodes", None)),
                                "has_edges": bool(result.get("result") and getattr(result.get("result"), "edges", None))
                            })
                        
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
                
                        # Log the LLM response structure
                        with self.logger.group("LLM Response Analysis"):
                            self.logger.info("Received WorkflowSpecLLM", data={
                                "reasoning_length": len(workflow_spec_llm.reasoning or ""),
                                "title": workflow_spec_llm.title,
                                "nodes_count": len(workflow_spec_llm.nodes) if workflow_spec_llm.nodes else 0,
                                "edges_count": len(workflow_spec_llm.edges) if workflow_spec_llm.edges else 0
                            })
                            
                            # Log data source nodes specifically
                            if workflow_spec_llm.nodes:
                                data_source_nodes = [n for n in workflow_spec_llm.nodes if n.type == "data_source"]
                                if data_source_nodes:
                                    self.logger.info("Data source nodes generated:", data={
                                        "count": len(data_source_nodes),
                                        "nodes": [{
                                            "label": n.label,
                                            "source_name": n.data.source_name,
                                            "has_config": bool(n.data.config),
                                            "config": n.data.config
                                        } for n in data_source_nodes]
                                    })
                
                        # Check if this is a chat-only response (nodes/edges are null)
                        if workflow_spec_llm.nodes is None or workflow_spec_llm.edges is None:
                            self.logger.info("Chat-only response detected", data={
                                "nodes": workflow_spec_llm.nodes,
                                "edges": workflow_spec_llm.edges,
                                "reasoning": workflow_spec_llm.reasoning if workflow_spec_llm.reasoning else None
                            })
                            # Return the WorkflowSpecLLM directly - no conversion needed
                            return workflow_spec_llm
                
                        # Convert LLM spec to final spec with deterministic IDs
                        workflow_spec = WorkflowSpec.from_llm_spec(workflow_spec_llm, uuid_for_workflow, rev=attempt)
                
                        # 🚨 CRITICAL VALIDATION: Check for tool hallucination and structural issues
                        with self.logger.group("Workflow Validation"):
                            # Build unified validation catalog (tools + data sources)
                            from ...utilities.tool_registry_utils import create_validation_catalog
                            
                            # Use the unified validation catalog function
                            validation_catalog = create_validation_catalog(
                                include_tools=bool(tool_catalog),
                                include_data_sources=True,
                                filter_broken=True,
                                verbose_format=False  # Use concise format for validation
                            )
                        
                            # Count data sources in the unified catalog
                            data_source_count = sum(1 for k, v in validation_catalog.items() 
                                                  if k in ['user_input', 'prompt_tool'])
                            
                            self.logger.info("Validation catalog prepared", data={
                                "tool_count": len(tool_catalog or {}) if tool_catalog else 0,
                                "data_source_count": data_source_count,
                                "total_catalog_size": len(validation_catalog)
                            })
                        
                            # Check validation bypass
                            if getattr(self, 'validation_bypass', False):
                                self.logger.warning("VALIDATION BYPASSED - Returning unvalidated workflow")
                                self.last_workflow = workflow_spec
                                return workflow_spec
                    
                            # Run validation
                            self.logger.info("Running structural validation")
                            structural_issues = workflow_spec.validate_structure(validation_catalog)
                        
                            if structural_issues:
                                self.logger.error(f"Validation failed with {len(structural_issues)} issues")
                                
                                # Log each issue
                                with self.logger.group("Validation Issues"):
                                    for i, issue in enumerate(structural_issues, 1):
                                        # Categorize the issue
                                        if "MISSING PARAMETERS" in issue:
                                            self.logger.critical(f"Issue {i}: {issue}")
                                        elif "EMPTY CONFIG" in issue:
                                            self.logger.critical(f"Issue {i}: {issue}")
                                        elif "HALLUCINATION" in issue:
                                            self.logger.error(f"Issue {i}: {issue}")
                                        else:
                                            self.logger.warning(f"Issue {i}: {issue}")
                                
                                if attempt <= max_retries:
                                    self.logger.info(f"Will retry with validation feedback (attempt {attempt + 1}/{max_retries + 1})")
                                    
                                    # Log the generated workflow for debugging
                                    self.logger.debug("Generated workflow that failed validation:", data={
                                        "workflow_prompt": workflow_spec.to_llm_prompt()
                                    })
                                    
                                    # Log validation issues to UI for transaction visibility
                                    validation_feedback = "VALIDATION ISSUES:\n" + "\n".join(f"{i}. {issue}" for i, issue in enumerate(structural_issues, 1))
                                    validation_feedback += f"\n\nGENERATED WORKFLOW:\n{workflow_spec.to_llm_prompt()}"
                                    
                                    log_prompt(
                                        prompt_type=f"validation_failure_attempt_{attempt}",
                                        prompt=validation_feedback,
                                        metadata={
                                            "attempt": attempt,
                                            "issues_count": len(structural_issues),
                                            "workflow_title": workflow_spec.title,
                                            "conversation_id": self.conversation_id,
                                            "retry_planned": True
                                        }
                                    )
                                    
                                    validation_errors.append(structural_issues)
                                    # CRITICAL: Pass the failed workflow to the prompt builder for next attempt
                                    prompt_builder.set_previous_workflow(workflow_spec)
                                    continue
                                else:
                                    error_msg = f"🚨 WORKFLOW VALIDATION FAILED AFTER {max_retries + 1} ATTEMPTS:\n" + "\n".join(structural_issues)
                                    error_msg += f"\n\nWorkflow reasoning: {workflow_spec.reasoning}"
                                    raise ValueError(error_msg)
                            else:
                                self.logger.success("Validation passed - workflow is structurally sound")
                
                        # Success! Store as last workflow for future context
                        self.logger.success(f"Workflow validated successfully on attempt {attempt}")
                        self.last_workflow = workflow_spec
                        return workflow_spec
                        
                    except ValueError:
                        raise  # Re-raise validation errors
                    except Exception as e:
                        # Handle other errors
                        self.logger.error(f"Error on attempt {attempt}: {str(e)}")
                        if attempt <= max_retries:
                            self.logger.info("Will retry after error")
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
                config = getattr(node.data, 'config', {})
                tool_nodes.append(f"  🔧 {node.label} (source: {tool_name}) config: {config}")
                
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