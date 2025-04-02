from typing import List, Dict, Any, Optional
import asyncio
import marvin

from .agents import Agent
from .utilities.helpers import LazyCaller


class Task:
    """
    A class to manage and orchestrate runs.
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
        context: Dict[str, Any] = None,
        result_type: Any = str,
        **kwargs,
    ) -> Any:
        """
        Wrap marvin.run() to execute a given objective

        :param objective: The primary task or objective to run.
        :param agents: A list of agents to use for this run. If None, uses self.agents.
        :param context: A dictionary of context data passed to the run.
        :param result_type: The expected return type (e.g. str, dict).
        :param kwargs: Additional keyword arguments passed directly to marvin.run().
        :return: The result of the marvin.run() call.
        """
        chosen_agents = agents if agents is not None else self.agents
        return LazyCaller(
            lambda: marvin.run(
                objective,
                agents=chosen_agents,
                context=context or {},
                result_type=result_type,
                **kwargs,
            )
        )

    async def a_run(
        self,
        objective: str,
        agents: List[Agent] = None,
        context: Dict[str, Any] = None,
        result_type: Any = str,
        **kwargs,
    ) -> Any:
        """
        Wrap marvin.run() to execute a given objective

        :param objective: The primary task or objective to run.
        :param agents: A list of agents to use for this run. If None, uses self.agents.
        :param context: A dictionary of context data passed to the run.
        :param result_type: The expected return type (e.g. str, dict).
        :param kwargs: Additional keyword arguments passed directly to marvin.run().
        :return: The result of the marvin.run() call.
        """
        chosen_agents = agents if agents is not None else self.agents
        # FIXME: could it break due to lambda returning awaitable? see if it needs fixes
        return LazyCaller(
            lambda: marvin.run_async(
                objective,
                agents=chosen_agents,
                context=context or {},
                result_type=result_type,
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
            "result_type": str
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
                    objective=spec["objective"],
                    agents=spec.get("agents"),
                    context=spec.get("context"),
                    result_type=spec.get("result_type", str),
                    **{
                        k: v
                        for k, v in spec.items()
                        if k not in ["objective", "agents", "context", "result_type"]
                    },
                )
                results.append(result)
            else:
                result = asyncio.run(
                    self.a_run(
                        objective=spec["objective"],
                        agents=spec.get("agents"),
                        context=spec.get("context"),
                        result_type=spec.get("result_type", str),
                        **{
                            k: v
                            for k, v in spec.items()
                            if k
                            not in ["objective", "agents", "context", "result_type"]
                        },
                    )
                )
                results.append(result)
        return results
