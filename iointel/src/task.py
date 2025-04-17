from typing import List, Dict, Any, Optional
import asyncio

from .agents import Agent
from .utilities.helpers import LazyCaller
from .agent_methods.data_models.datamodels import TaskDefinition


class Task:
    """
    A class to manage and orchestrate runs.
    It can store a set of agents and provide methods to run them with given instructions and context.
    """

    def __init__(self, agents: List[Agent] = None):
        """
        :param agents: Optional list of Agent instances that this runner can orchestrate.
        """
        self.agents = agents or []
        self.current_agent_idx = 0

    def add_agent(self, agent: Agent):
        """
        Add a new agent to the runner's collection.
        """
        self.agents.append(agent)

    def get_next_agent(self) -> Agent:
        if not self.agents:
            raise ValueError("No agents available to run the task")
        agent = self.agents[self.current_agent_idx]
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)
        return agent

    async def run_stream(self, definition: TaskDefinition, **kwargs) -> Any:
        chosen_agents = definition.agents or self.agents
        if not chosen_agents:
            raise ValueError("No agents available for this task.")

        # Do NOT assign to self.agents; instead select locally
        active_agent = chosen_agents[0]

        output_type = kwargs.pop("output_type", str)

        return LazyCaller(
            lambda: active_agent.run_stream(
                query=definition.objective,
                conversation_id=definition.task_metadata.get("conversation_id")
                if definition.task_metadata
                else None,
                output_type=output_type,
                **kwargs,
            )
        )

    def run(self, definition: TaskDefinition, **kwargs) -> Any:
        """
        Synchronous run. You might want to unify the parameters
        so that you pass a single TaskDefinition instead of
        separate objective/agents/etc.
        """

        chosen_agents = definition.agents or self.agents
        if chosen_agents:
            self.agents = chosen_agents

        active_agent = self.get_next_agent()
        # active_agent.output_type = (kwargs.get("output_type")
        #                            or str)

        output_type = kwargs.pop("output_type", str)

        return LazyCaller(
            lambda: active_agent.run_sync(
                query=definition.objective,
                conversation_id=definition.task_metadata.get("conversation_id")
                if definition.task_metadata
                else None,
                output_type=output_type,
                **kwargs,
            )
        )

    async def a_run(self, definition: TaskDefinition, **kwargs) -> Any:
        chosen_agents = definition.agents or self.agents
        if chosen_agents:
            self.agents = chosen_agents

        active_agent = self.get_next_agent()
        # active_agent.output_type = (kwargs.get("output_type")
        #                            or str)
        output_type = kwargs.pop("output_type", str)
        return LazyCaller(
            lambda: active_agent.run_async(
                query=definition.objective,
                conversation_id=definition.task_metadata.get("conversation_id")
                if definition.task_metadata
                else None,
                output_type=output_type,
                **kwargs,
            )
        )

    def chain_runs(
        self, run_specs: List[Dict[str, Any]], run_async: Optional[bool] = False
    ) -> List[Any]:
        """
        Execute multiple runs in sequence. Each element in run_specs is a dict containing parameters for `self.run`.
        The output of one run can be fed into the context of the next run if desired.

        Example run_specs:
        [
          {
            "objective": "Deliberate on task",
            "instructions": "...",
            "output_type": str
          },
          {
            "objective": "Use the result of the previous run to code a solution",
            "instructions": "...",
            "context": {"previous_result": "$0"}  # '$0' means use the result of the first run
          }
        ]

        :param run_specs: A list of dictionaries, each describing one run's parameters.
        :return: A list of results from each run in order.
        """
        results = []
        for i, spec in enumerate(run_specs):
            # Resolve any placeholders in context using previous results
            context = spec.get("context", {})
            if context:
                resolved_context = {}
                for k, v in context.items():
                    if isinstance(v, str) and v.startswith("$"):
                        # Format: "$<index>" to reference a previous run's result
                        idx = int(v[1:])
                        resolved_context[k] = results[idx]
                    else:
                        resolved_context[k] = v
                spec["context"] = resolved_context

            if not run_async:
                # Execute the run
                result = self.run(
                    query=spec["objective"],
                    agents=spec.get("agents"),
                    context=spec.get("context"),
                    conversation_id=spec.get("conversation_id"),
                    **{
                        k: v
                        for k, v in spec.items()
                        if k
                        not in ["objective", "agents", "context", "conversation_id"]
                    },
                )
                results.append(result)
            else:
                result = asyncio.run(
                    self.a_run(
                        query=spec["objective"],
                        agents=spec.get("agents"),
                        context=spec.get("context"),
                        output_type=spec.get("output_type", str),
                        conversation_id=spec.get("conversation_id"),
                        **{
                            k: v
                            for k, v in spec.items()
                            if k
                            not in ["objective", "agents", "context", "output_type"]
                        },
                    )
                )
                results.append(result)
        return results
