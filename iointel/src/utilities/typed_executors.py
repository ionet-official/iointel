"""Typed executors for different node types."""

from typing import Any, Optional, List
from .typed_execution import TypedExecutionContext
from ..utilities.io_logger import get_component_logger
from ..agent_methods.data_models.datamodels import AgentResultFormat, AgentParams
from ..agents import Agent
from ..agent_methods.agents.agents_factory import create_agent

logger = get_component_logger("TYPED_EXECUTORS")


async def execute_agent_typed(context: TypedExecutionContext) -> Any:
    """
    Execute an agent node with typed context.
    
    This replaces the dict-based execute_agent_task.
    """
    from ..utilities.runners import run_agents
    
    # Get or create agents
    agents = context.agents or []
    if not agents and context.agent_instructions:
        # Create agent from node data
        agent = await _create_agent_from_context(context)
        if agent:
            agents = [agent]
    
    if not agents:
        logger.warning(f"No agents available for node {context.node_id}")
        return {"error": "No agents available"}
    
    # Build context from available results using edge information
    agent_context = {}
    
    # Use edge-based data flow from the workflow spec
    input_data = context.get_input_data()
    
    # Determine objective (query) and context separately
    objective = None
    
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
            logger.info(f"Multiple inputs, creating descriptive objective for processing")
        
        # ALL input data goes into context (including the objective)
        agent_context = input_data
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
    
    # Execute agent
    response = await run_agents(
        objective=objective,
        agents=agents,
        context=agent_context,
        conversation_id=context.conversation_id,
        output_type=str,
        result_format=result_format
    ).execute()
    
    return response


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
    
    # Resolve variables in config
    config = context.resolve_config()
    
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
        request = DataSourceRequest(
            message=config.get("message", ""),
            default_value=config.get("default_value")
        )
    else:
        # For other data sources, pass config directly
        request = config
    
    # Execute data source
    execution_metadata = {
        "node_id": context.node_id,
        "conversation_id": context.conversation_id,
        # Pass through any user_inputs from the state
        "user_inputs": context.state.user_inputs if hasattr(context.state, 'user_inputs') else {}
    }
    
    response = data_source_func(request, execution_metadata=execution_metadata)
    
    # Extract result value
    if hasattr(response, 'result'):
        return response.result
    elif hasattr(response, 'message'):
        return response.message
    else:
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
            model=config["model_config"]["model"],
            tools=context.node_data.tools or [],
            api_key=config["api_config"]["api_key"] if "api_key" in config.get("api_config", {}) else None,
            api_type=config["api_config"]["api_type"],
            api_base=config["api_config"].get("api_base"),
            model_kwargs=config["model_config"]["model_kwargs"]
        )
        
        # Create agent
        agent = create_agent(agent_params)
        logger.info(f"Created agent for node {context.node_id} with static instructions")
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create agent for node {context.node_id}: {e}")
        return None