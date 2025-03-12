from typing import List, Dict, Any, Optional
from .agents import Agent
import marvin
import asyncio


class Task:
    """
    A class to manage and orchestrate runs using Marvin's marvin.run().
    It can store a set of agents and provide methods to run them with given instructions and context.
    """

    def __init__(self, agents: List[Agent] = None):
        """
        :param agents: Optional list of Agent instances that this runner can orchestrate.
        """
        self.agents = agents

    def add_agent(self, agent: Agent):
        """
        Add a new agent to the runner's collection.
        """
        self.agents.append(agent)

    def run(
        self,
        objective: str,
        agents: List[Agent] = None,
        result_type: Any = str,
        **kwargs,
    ) -> Any:
        """
        Wrap marvin.run() to execute a given objective

        :param objective: The primary task or objective to run.
        :param agents: A list of agents to use for this run. If None, uses self.agents.
        :param result_type: The expected return type (e.g. str, dict).
        :param kwargs: Additional keyword arguments passed directly to mavrin.run().
        :return: The result of the marvin.run() call.
        """
        chosen_agents = agents if agents is not None else self.agents
        return marvin.run(
            instructions=objective,
            agents=chosen_agents,
            result_type=result_type,
            **kwargs,
        )

    async def a_run(
        self,
        objective: str,
        agents: List[Agent] = None,
        result_type: Any = str,
        **kwargs,
    ) -> Any:
        """
        Wrap marvin.run() to execute a given objective

        :param objective: The primary task or objective to run.
        :param agents: A list of agents to use for this run. If None, uses self.agents.
        :param result_type: The expected return type (e.g. str, dict).
        :param kwargs: Additional keyword arguments passed directly to mavrin.run().
        :return: The result of the marvin.run() call.
        """
        chosen_agents = agents if agents is not None else self.agents
        return await marvin.run(
            instructions=objective,
            agents=chosen_agents,
            result_type=result_type,
            **kwargs,
        )

# A global or module-level registry of custom workflows
CUSTOM_WORKFLOW_REGISTRY = {}


def register_custom_workflow(name: str):
    def decorator(func):
        CUSTOM_WORKFLOW_REGISTRY[name] = func
        return func

    return decorator
