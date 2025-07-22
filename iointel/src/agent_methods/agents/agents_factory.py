from pydantic import BaseModel
from ...agents import Agent
from ..data_models.datamodels import AgentParams, Tool, AgentSwarm
from typing import Callable, Sequence
from .tool_factory import instantiate_stateful_tool, resolve_tools
from ..data_models.agent_pre_prompt_injection import inject_pre_prompts
from ...utilities.helpers import make_logger

logger = make_logger(__name__)


def instantiate_agent_default(params: AgentParams) -> Agent:
    return Agent(**params.model_dump(exclude={"tools", "sla_requirements"}), tools=params.tools)


def create_agent(
    params: AgentParams,
    instantiate_agent: Callable[[AgentParams], Agent] | None = None,
    instantiate_tool: Callable[[Tool, dict | None], BaseModel | None] | None = None,
) -> Agent:
    """
    Create an Agent instance from the given AgentParams.
    When rehydrating from YAML, each tool in params.tools is expected to be either:
      - a string - tool name,
      - a pair of (string, dict) - tool name + args to reconstruct tool self,
      - a dict (serialized Tool) with a "body" field,
      - a Tool instance,
      - or a callable.
    In the dict case, we ensure that the "body" is preserved.

    The `instantiate_tool` is called when a tool is "stateful",
    i.e. it is an instancemethod, and its `self` is not yet initialized,
    and its purpose is to allow to customize the process of Tool instantiation.

    The `instantiate_agent` is called to create `Agent` from `AgentParams`,
    and its purpose is to allow to customize the process of Agent instantiation.
    """
    # Dump the rest of the agent data (excluding tools) then reinsert our resolved tools.
    tools = resolve_tools(
        params,
        tool_instantiator=instantiate_stateful_tool
        if instantiate_tool is None
        else instantiate_tool,
    )
    
    # Extract tool names for classification
    tool_names = []
    for tool in params.tools:
        if isinstance(tool, str):
            tool_names.append(tool)
        elif hasattr(tool, 'name'):
            tool_names.append(tool.name)
        elif isinstance(tool, dict) and 'name' in tool:
            tool_names.append(tool['name'])
    
    # Determine effective SLA requirements from AgentParams
    effective_sla = params.sla_requirements
    # If not set, fallback to catalog/classifier (None means use default logic in inject_pre_prompts)
    
    # Apply pre-prompt injection based on agent type classification and SLA
    enhanced_instructions, classification = inject_pre_prompts(
        original_instructions=params.instructions,
        tools=tool_names,
        agent_name=params.name,
        context=None,
        sla_requirements=effective_sla
    )
    
    logger.info(f"🤖 Agent '{params.name}' classified as {classification.agent_type.value} "
                f"(confidence: {classification.confidence:.2f})")
    logger.debug(f"   Reasoning: {classification.reasoning}")
    if classification.sla_enforcement:
        logger.info("   🔒 SLA enforcement enabled")
    
    output_type = params.output_type
    if isinstance(output_type, str):
        output_type = globals().get(output_type) or __builtins__.get(
            output_type, output_type
        )
    
    # Create agent with enhanced instructions (keep SLA in AgentParams, exclude from Agent)
    enhanced_params = params.model_copy(update={
        "tools": tools, 
        "output_type": output_type,
        "instructions": enhanced_instructions,
        "sla_requirements": effective_sla  # Keep in AgentParams for SLA system
    })
    
    return (
        instantiate_agent_default if instantiate_agent is None else instantiate_agent
    )(enhanced_params)


def create_swarm(agents: list[AgentParams] | AgentSwarm):
    raise NotImplementedError()


def agent_or_swarm(
    agent_obj: Agent | Sequence[Agent], store_creds: bool
) -> list[AgentParams] | AgentSwarm:
    """
    Serializes an agent object into a list of AgentParams.

    - If the agent_obj is an individual agent (has an 'api_key'),
      returns a list with one AgentParams instance.
    - If the agent_obj is a swarm (has a 'members' attribute),
      returns a list of AgentParams for each member.
    """

    def get_api_key(agent: Agent) -> str:
        if not (api_key := agent.api_key):
            return None
        if store_creds and hasattr(api_key, "get_secret_value"):
            return api_key.get_secret_value()
        return api_key

    def make_params(agent: Agent) -> AgentParams:
        return AgentParams(
            name=agent.name,
            instructions=agent.instructions,
            persona=agent.persona,
            tools=[
                Tool.from_function(t).model_dump(exclude={"fn", "fn_metadata"})
                for t in agent.tools
            ],
            model=getattr(agent.model, "model_name", None),
            model_settings=agent.model_settings,
            api_key=get_api_key(agent),
            base_url=agent.base_url,
            memory=agent.memory,
            context=agent.context,
            output_type=agent.output_type,
        )

    if isinstance(agent_obj, Sequence):
        # group of agents not packed as a swarm
        assert all(not hasattr(ag, "members") for ag in agent_obj), (
            "Nested swarms not allowed"
        )
        return [make_params(ag) for ag in agent_obj]
    if hasattr(agent_obj, "api_key"):
        # Individual agent.
        return [make_params(agent_obj)]
    if hasattr(agent_obj, "members"):
        # Swarm: return AgentParams for each member.
        return AgentSwarm(members=[make_params(member) for member in agent_obj.members])
    # Fallback: return a minimal AgentParams.
    return [make_params(agent_obj)]
