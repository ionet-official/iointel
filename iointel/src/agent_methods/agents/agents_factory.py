from pydantic import BaseModel
from iointel.src.agents import Agent
from iointel.src.agent_methods.data_models.workflow_spec import SLARequirements
from iointel.src.agent_methods.data_models.datamodels import AgentParams, Tool, AgentSwarm
from typing import Callable, Sequence
from iointel.src.agent_methods.agents.tool_factory import instantiate_stateful_tool, resolve_tools
from iointel.src.agent_methods.data_models.agent_pre_prompt_injection import inject_prompts_enforcement_from_sla
from iointel.src.utilities.io_logger import get_component_logger

logger = get_component_logger("AGENTS-FACTORY")


def instantiate_agent_default(params: AgentParams) -> Agent:
    # HACK: Force correct API configuration for Llama models
    from ...utilities.constants import get_model_config
    
    agent_dict = params.model_dump(exclude={"tools", "sla_requirements"})
    
    # Override API configuration based on model
    if agent_dict.get("model"):
        print(f"🔧 HACK: Fixing API config for model: {agent_dict['model']}")
        print(f"   Original api_key: {agent_dict.get('api_key', 'None')[:10]}..." if agent_dict.get('api_key') else "   Original api_key: None")
        print(f"   Original base_url: {agent_dict.get('base_url', 'None')}")
        
        config = get_model_config(
            model=agent_dict["model"],
            api_key=agent_dict.get("api_key"),
            base_url=agent_dict.get("base_url")
        )
        agent_dict["model"] = config["model"]
        agent_dict["api_key"] = config["api_key"]
        agent_dict["base_url"] = config["base_url"]
        
        print(f"   New api_key: {config['api_key'][:10]}..." if config['api_key'] else "   New api_key: None")
        print(f"   New base_url: {config['base_url']}")
    
    return Agent(**agent_dict, tools=params.tools)


def create_agent(
    params: AgentParams,
    instantiate_agent: Callable[[AgentParams], Agent] | None = None,
    instantiate_tool: Callable[[Tool, dict | None], BaseModel | None] | None = None,
    sla_requirements: SLARequirements | None = None,
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

    The `sla_requirements` is a `SLARequirements` object,
    and it is used to inject pre-prompts into the agent's instructions.
    It is also used to validate the agent's tool usage.
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
    
    
    # Apply pre-prompt injection based on SLA requirements (single source of truth)
    # TODO: Get SLA from NodeSpec when available, for now use legacy classification
    enhanced_instructions = inject_prompts_enforcement_from_sla(
        original_instructions=params.instructions,
        sla_requirements=sla_requirements,
    )
    
    logger.info(f"🤖 Agent '{params.name}' instructions enhanced")
    if sla_requirements:
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
        "sla_requirements": sla_requirements  # Keep in AgentParams for SLA system
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
                # agent.tools already contains Tool objects, not functions
                t.model_dump(exclude={"fn", "fn_metadata"}) if isinstance(t, Tool) else t
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
