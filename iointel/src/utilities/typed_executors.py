"""Typed executors for different node types."""

from typing import Any, Optional
from iointel.src.utilities.typed_execution import TypedExecutionContext
from iointel.src.utilities.io_logger import get_component_logger, log_prompt
from iointel.src.agent_methods.data_models.datamodels import AgentResultFormat, AgentParams
from iointel.src.agent_methods.data_models.execution_models import AgentExecutionResult, ExecutionStatus, AgentRunResponse
from iointel.src.agents import Agent
from iointel.src.agent_methods.agents.agents_factory import create_agent
from iointel.src.agent_methods.data_models.agent_pre_prompt_injection import inject_prompts_enforcement_from_sla

logger = get_component_logger("TYPED_EXECUTORS")


async def execute_agent_typed(context: TypedExecutionContext) -> Any:
    """
    Execute an agent node with typed context.
    
    This replaces the dict-based execute_agent_task.
    """
    from ..utilities.runners import run_agents
    from ..utilities.node_execution_wrapper import sla_validator
    
    # Always create agent from the WorkflowSpec
    agents = []
    
    if context.agent_instructions:
        logger.info(f"Creating agent from WorkflowSpec for node {context.node_id}")
        logger.info(f"  Instructions: {context.agent_instructions[:100]}...")
        logger.info(f"  Tools: {context.node_data.tools}")
        
        agent = await _create_agent_from_context(context)
        if agent:
            agents = [agent]
            logger.info(f"Successfully created agent: {agent.name}")
        else:
            logger.error(f"Failed to create agent from context for node {context.node_id}")
    else:
        logger.error(f"Node {context.node_id} has no agent_instructions in WorkflowSpec!")
    
    if not agents:
        logger.error(f"No agents available for node {context.node_id}")
        return {"error": "No agents available"}
    
    # Build context from available results using edge information
    agent_context = {}
    
    # Use edge-based data flow from the workflow spec
    input_data = context.get_input_data()
    
    # Determine objective (query) and context separately
    objective = None
    
    # BACKWARD COMPATIBILITY: If node has empty ins[] but has incoming edges,
    # pass ALL available results as context (old behavior)
    node_ins = getattr(context.node_data, 'ins', [])
    if (not node_ins or len(node_ins) == 0) and context.incoming_edges:
        logger.info(f"Node {context.current_node_id} has empty ins[] but has edges - using ALL results for backward compatibility")
        # Pass all available results as context
        agent_context = context.available_results.copy()
        # Still get edge-based input data for objective determination
        if not input_data and agent_context:
            # Use first string value as objective if available
            for key, value in agent_context.items():
                if isinstance(value, str):
                    objective = value
                    logger.info(f"Using {key} as objective from available results: {objective[:50]}...")
                    break
    
    # If we have edge-based input data, determine what's the query vs context
    if input_data:
        # Strategy: If there's a single string input, use it as the objective
        # All other inputs (including the string) go into context
        string_inputs = [(k, v) for k, v in input_data.items() if isinstance(v, str)]
        
        if len(string_inputs) == 1:
            # Single string input - use as objective
            source_id, value = string_inputs[0]
            objective = value
            logger.info(f"Using input from {source_id} as objective: {objective[:50]}...")
        elif len(string_inputs) > 1:
            # Multiple string inputs - create a descriptive objective
            objective = f"Process the values in your context: {', '.join(input_data.keys())}"
            logger.info("Multiple inputs, creating descriptive objective for processing")
        
        # ALL input data goes into context (including the objective)
        # Merge with any existing context from backward compatibility
        agent_context.update(input_data)
        logger.info(f"Node {context.current_node_id} receiving inputs from edges: {list(input_data.keys())}")
    elif context.get_dependencies():
        # Node has dependencies but no data yet - this shouldn't happen in normal flow
        logger.warning(f"Node {context.current_node_id} has dependencies {context.get_dependencies()} but no input data")
    
    # Log what context the agent is receiving
    if agent_context:
        logger.info(f"Agent context for {context.current_node_id}: {list(agent_context.keys())}")
    
    # Final fallback if objective is still None
    if not objective:
        if context.objective:
            objective = context.objective
        else:
            objective = "Execute the task as instructed"
            logger.info(f"No objective specified for {context.current_node_id}, using default")
    
    # Determine result format
    result_format = AgentResultFormat.full()
    if context.node_type == "decision":
        result_format = AgentResultFormat.workflow()
    
    logger.info(f"Executing agent {agents[0].name if agents else 'unknown'}")
    logger.debug(f"Objective: {objective[:100] if objective else 'None'}...")
    logger.debug(f"Context keys: {list(agent_context.keys())}")
    
    # Execute agent and return typed result
    import time
    
    start_time = time.time()
    
    # Log the prompt before execution
    prompt_id = log_prompt(
        prompt_type="agent_instruction",
        prompt=objective,
        metadata={
            "node_id": context.current_node_id,
            "agent_name": agents[0].name if agents else "unknown",
            "node_label": context.current_node.label,
            "node_type": context.node_type,
            "context_keys": list(agent_context.keys()),
            "sla_requirements": context.current_node.sla
        }
    )
    
    # Define the execution function for SLA validation
    async def execute_fn():
        try:
            # Execute the agent
            response = await run_agents(
                objective=objective,
                agents=agents,
                context=agent_context,
                conversation_id=context.conversation_id,
                output_type=str,
                result_format=result_format
            ).execute()
            
            # Convert dict response to typed AgentRunResponse
            if isinstance(response, dict):
                typed_response = AgentRunResponse.from_dict(response)
            else:
                # Already typed response
                typed_response = response
            
            # Log the response after execution
            if typed_response and typed_response.result:
                log_prompt(
                    prompt_type="agent_response",
                    prompt=objective,
                    response=str(typed_response.result)[:2000],  # Limit response length
                    metadata={
                        "node_id": context.current_node_id,
                        "agent_name": agents[0].name if agents else "unknown",
                        "prompt_id": prompt_id,  # Link to original prompt
                        "tool_usage": [t.tool_name for t in typed_response.tool_usage_results] if typed_response.tool_usage_results else [],
                        "execution_time": time.time() - start_time
                    }
                )
            
            # Return typed execution result - this matches what execute_agent_task returns
            return AgentExecutionResult(
                status=ExecutionStatus.COMPLETED,
                agent_response=typed_response,
                execution_time=time.time() - start_time,
                node_id=context.node_id
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Return typed error result
            return AgentExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                node_id=context.node_id
            )
    
    # Execute with SLA validation if this is a decision node or has SLA requirements
    if context.node_type == "decision" or context.current_node.sla:
        logger.debug(f"Executing {context.node_type} node {context.node_id} with SLA validation")
        return await sla_validator.validate_async(
            execute_fn=execute_fn,
            node_spec=context.current_node,
            allow_retry=True
        )
    else:
        # No SLA validation needed
        return await execute_fn()


async def execute_data_source_typed(context: TypedExecutionContext) -> Any:
    """
    Execute a data source node with typed context.
    """
    from ..agent_methods.data_sources import get_data_source
    from ..agent_methods.data_sources.models import DataSourceRequest
    
    source_name = context.node_data.source_name
    if not source_name:
        logger.error(f"No source_name for data source node {context.node_id}")
        return {"error": "No source_name specified"}
    
    # Get config - it's now a DataSourceConfig Pydantic model for data_source nodes
    from ..agent_methods.data_models.workflow_spec import DataSourceConfig
    config = context.node_data.config
    
    logger.info(f"Executing data source '{source_name}'")
    logger.debug(f"Config: {config}")
    
    # Get data source function
    try:
        data_source_func = get_data_source(source_name)
    except KeyError:
        logger.error(f"Data source '{source_name}' not found")
        return {"error": f"Data source '{source_name}' not found"}
    
    # Create request based on source type
    if source_name == "user_input":
        # Config is a DataSourceConfig with message and default_value fields
        if isinstance(config, DataSourceConfig):
            request = DataSourceRequest(
                message=config.message,
                default_value=config.default_value
            )
        else:
            # Fallback for legacy dict config
            request = DataSourceRequest(
                message=config.get("message", "") if isinstance(config, dict) else "",
                default_value=config.get("default_value") if isinstance(config, dict) else None
            )
    else:
        # For other data sources, pass config directly
        request = config
    
    # Execute data source with runtime metadata
    execution_metadata = {
        "node_id": context.node_id,
        "conversation_id": context.conversation_id,
        # Get user_inputs directly from state where they belong
        "user_inputs": context.state.user_inputs
    }
    
    response = data_source_func(request, execution_metadata=execution_metadata)
    
    # Extract result value based on response type
    from ..agent_methods.data_sources.models import DataSourceResponse
    
    if isinstance(response, DataSourceResponse):
        # DataSourceResponse has 'message' field for the actual content
        return response.message or response.default_value
    else:
        # Fallback for other response types
        return response


async def execute_decision_typed(context: TypedExecutionContext) -> Any:
    """
    Execute a decision node with typed context.
    
    Decision nodes are just agents with special result formatting.
    """
    # Ensure we use workflow format for decisions
    original_type = context.node_type
    context.node.type = "decision"  # Ensure execute_agent_typed knows this is a decision
    
    try:
        return await execute_agent_typed(context)
    finally:
        context.node.type = original_type


async def execute_tool_typed(context: TypedExecutionContext) -> Any:
    """
    Execute a tool node with typed context.
    """
    # This would integrate with the tool registry
    # For now, raise NotImplementedError
    logger.error(f"Tool node execution not yet implemented for {context.node_id}")
    raise NotImplementedError("Tool node execution not yet implemented in typed executor")


async def _create_agent_from_context(context: TypedExecutionContext) -> Optional[Agent]:
    """
    Create an agent from the node context.
    
    This replaces the logic in _hydrate_agents_from_node.
    """
    if not context.agent_instructions:
        return None
    
    try:
        from ..utilities.constants import get_model_config
        
        # Get model configuration
        config = get_model_config(
            model=context.node_data.model or "gpt-4o",
            api_key=None  # Use defaults
        )
        
        # Create agent parameters
        # IMPORTANT: Agent instructions are static header instructions, NOT templated
        agent_params = AgentParams(
            name=f"agent_{context.node_id}",
            instructions=context.agent_instructions,  # Static instructions, no variable resolution
            model=config["model"],
            tools=context.node_data.tools or [],
            api_key=config.get("api_key")
            # Don't pass api_type, api_base, or model_kwargs - they cause issues with OpenAIModel
            # These are determined internally based on the model
        )
        
        # Pass node type and SLA to agent factory through a custom creation function
        # that can inject the proper pre-prompts based on node type
        
        # Get SLA requirements from NodeSpec
        sla_requirements = context.current_node.sla
        
        # If this is a decision node or has SLA requirements, enhance instructions
        if context.node_type == "decision" or (sla_requirements and sla_requirements.tool_usage_required):
            # For decision nodes, add route mapping information
            instructions_to_enhance = context.agent_instructions
            if context.node_type == "decision":
                # Get outgoing edges to understand route mappings
                outgoing_edges = context.outgoing_edges
                if outgoing_edges:
                    route_info = ["\n### ROUTING CONFIGURATION ###"]
                    route_info.append("When calling conditional_gate, use these exact route configurations:")
                    for edge in outgoing_edges:
                        if edge.data.route_index is not None:
                            target_node = next((n for n in context.workflow_spec.nodes if n.id == edge.target), None)
                            target_label = target_node.label if target_node else edge.target
                            route_info.append(f"  - route_index: {edge.data.route_index} → '{edge.data.route_label or target_label}' (goes to {target_label})")
                    route_info.append("IMPORTANT: Call conditional_gate ONLY ONCE with the appropriate route_index based on your analysis.")
                    route_info.append("### END ROUTING CONFIGURATION ###\n")
                    instructions_to_enhance = context.agent_instructions + "\n" + "\n".join(route_info)
            
            # Use inject_pre_prompts_from_sla which handles None sla_requirements gracefully
            enhanced_instructions = inject_prompts_enforcement_from_sla(
                original_instructions=instructions_to_enhance,
                sla_requirements=sla_requirements
            )
            agent_params.instructions = enhanced_instructions
            logger.info(f"Enhanced agent instructions for {context.node_type} node {context.node_id} with SLA enforcement")
        
        # Create agent with enhanced instructions
        agent = create_agent(agent_params)
        logger.info(f"Created agent for node {context.node_id} (type: {context.node_type})")
        
        return agent
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to create agent for node {context.node_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None