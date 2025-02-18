from framework import Agent, Swarm
from ..data_models.datamodels import AgentParams
from typing import List
from .tool_factory import resolve_tools
import logging


import logging
import os
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Fallback to "DEBUG" if not set
level_name = os.environ.get("AGENT_LOGGING_LEVEL", "INFO")
level_name = level_name.upper()
# Safely get the numeric logging level, default to DEBUG if invalid
numeric_level = getattr(logging, level_name, logging.INFO)
logger.setLevel(numeric_level)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



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
    #agent_data["tools"] = [tool.fn for tool in resolved_tools]
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
        if store_creds and hasattr(agent.api_key, "get_secret_value"):
            return agent.api_key.get_secret_value()
        return agent.api_key

    if hasattr(agent_obj, "api_key"):
        # Individual agent.
        return [
            AgentParams(
                name=agent_obj.name,
                instructions=agent_obj.instructions,
                tools=[Tool.from_function(t).model_dump(exclude={"fn", "fn_metadata"})
                       for t in agent_obj.tools],
                model=getattr(agent_obj.model, "model_name", None),
                model_settings=agent_obj.model_settings,
                api_key=get_api_key(agent_obj),
                base_url=agent_obj.base_url,
                memories=agent_obj.memories,
            )
        ]
    elif hasattr(agent_obj, "members"):
        # Swarm: return AgentParams for each member.
        return [
            AgentParams(
                name=member.name,
                instructions=member.instructions,
                tools=[Tool.from_function(t).model_dump(exclude={"fn", "fn_metadata"})
                       for t in member.tools],
                model=getattr(member.model, "model_name", None),
                model_settings=member.model_settings,
                api_key=get_api_key(member),
                base_url=member.base_url,
                memories=member.memories,
                swarm_name=agent_obj.name,
            )
            for member in agent_obj.members
        ]
    else:
        # Fallback: return a minimal AgentParams.
        return [
                AgentParams(
                    name=agent_obj.name, 
                    instructions=agent_obj.instructions, 
                    tools=[Tool.from_function(t).model_dump(exclude={"fn", "fn_metadata"})
                            for t in agent_obj.tools], 
                    model=getattr(agent_obj.model, "model_name", None), 
                    model_settings=agent_obj.model_settings, 
                    api_key=get_api_key(agent_obj), 
                    base_url=agent_obj.base_url, memories=agent_obj.memories
                )
            ]