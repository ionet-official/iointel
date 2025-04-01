from ...agents import Agent, Swarm
from ..data_models.datamodels import AgentParams
from typing import List
from .tool_factory import resolve_tools


def create_agent(params: AgentParams) -> Agent:
    """
    Create an Agent instance from the given AgentParams.
    When rehydrating from YAML, each tool in params.tools is expected to be either:
      - a dict (serialized Tool) with a "body" field,
      - a Tool instance,
      - or a callable.
    In the dict case, we ensure that the "body" is preserved.
    """
    # Dump the rest of the agent data (excluding tools) then reinsert our resolved tools.
    agent_data = params.model_dump(exclude={"tools"})
    agent_data["tools"] = resolve_tools(params)
    return Agent(**agent_data)


def create_swarm(agents: List[Agent]) -> Swarm:
    return Swarm(agents)


def agent_or_swarm(agent_obj, store_creds: bool) -> list:
    """
    Serializes an agent object into a list of AgentParams.
    
    - If the agent_obj is an individual agent (has an 'api_key'),
      returns a list with one AgentParams instance.
    - If the agent_obj is a swarm (has a 'members' attribute),
      returns a list of AgentParams for each member.
    """
    from ...agent_methods.data_models.datamodels import AgentParams, Tool

    def get_api_key(agent):
        if (api_key := getattr(agent, "api_key", None)) is None:
            return None
        if store_creds and hasattr(api_key, "get_secret_value"):
            return api_key.get_secret_value()
        return api_key

    def make_params(agent):
        return AgentParams(
            name=agent.name,
            instructions=agent.instructions,
            tools=[Tool.from_function(t).model_dump(exclude={"fn", "fn_metadata"}) for t in agent.tools],
            model=getattr(agent.model, "model_name", None),
            model_settings=agent.model_settings,
            api_key=get_api_key(agent),
            base_url=agent.base_url,
            memories=agent.memories,
        )

    if hasattr(agent_obj, "api_key"):
        # Individual agent.
        return [make_params(agent_obj)]
    elif hasattr(agent_obj, "members"):
        # Swarm: return AgentParams for each member.
        return [make_params(member) for member in agent_obj.members]
    else:
        # Fallback: return a minimal AgentParams.
        return [make_params(agent_obj)]
