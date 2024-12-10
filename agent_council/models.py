from langchain_openai import ChatOpenAI
import controlflow as cf
import os
from typing import List, Dict, Any


class Agent(cf.Agent):

    """
    A configurable wrapper around cf.Agent that allows you to plug in different chat models, 
    instructions, and tools. By default, it uses the ChatOpenAI model.
    
    In the future, you can add logic to switch to a Llama-based model or other models by 
    adding conditionals or separate model classes.
    """


    def __init__(
        self,
        name: str,
        instructions: str,
        tools: list = None,
        model_provider: callable = None,
        **model_kwargs
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param tools: A list of cf.Tool instances or @cf.tool decorated functions.
        :param model_provider: A callable that returns a configured model instance. 
                              If provided, it should handle all model-related configuration.
        :param model_kwargs: Additional keyword arguments passed to the model factory or ChatOpenAI if no factory is provided.
        
        If model_provider is given, you rely entirely on it for the model and ignore other model-related kwargs.
        If not, you fall back to ChatOpenAI with model_kwargs such as model="gpt-4o-mini", api_key="..."
        """

        if model_provider is not None:
            model = model_provider(**model_kwargs)
        else:
            model = ChatOpenAI(**model_kwargs)

        # Call super with all required fields
        super().__init__(
            name=name,
            instructions=instructions,
            tools=tools or [],
            model=model,
        )


    def run(self, prompt: str):
        # Since you're subclassing cf.Agent, you can call super().run directly.
        return super().run(prompt)

    def set_instructions(self, new_instructions: str):
        # Update the instructions field on the pydantic model using assignment
        # This should be done through a mechanism that cf.Agent supports, or 
        # ensure that cf.Agent fields are defined as mutable. If it causes errors, 
        # you may need to recreate the agent or rely on internal CF methods.
        self.instructions = new_instructions

    def add_tool(self, tool):
        # Append to tools the same way. If it's a field managed by pydantic, 
        # ensure it is allowed. Otherwise, consider recreating the agent.
        updated_tools = self.tools + [tool]
        self.tools = updated_tools


class AgentRunner:
    """
    A class to manage and orchestrate runs using ControlFlow's cf.run().
    It can store a set of agents and provide methods to run them with given instructions and context.
    """

    def __init__(self, agents: List[cf.Agent] = None):
        """
        :param agents: Optional list of cf.Agent instances that this runner can orchestrate.
        """
        self.agents = agents or []

    def add_agent(self, agent: cf.Agent):
        """
        Add a new agent to the runner's collection.
        """
        self.agents.append(agent)

    def run(
        self,
        objective: str,
        agents: List[cf.Agent] = None,
        completion_agents: List[cf.Agent] = None,
        instructions: str = "",
        context: Dict[str, Any] = None,
        result_type: Any = str,
        **kwargs
    ) -> Any:
        """
        Wrap cf.run() to execute a given objective with optional instructions, context, and agents.

        :param objective: The primary task or objective to run.
        :param agents: A list of agents to use for this run. If None, uses self.agents.
        :param completion_agents: Agents that finalize the run (e.g., selecting a final answer).
        :param instructions: Additional instructions or prompt details for the run.
        :param context: A dictionary of context data passed to the run.
        :param result_type: The expected return type (e.g. str, dict).
        :param kwargs: Additional keyword arguments passed directly to cf.run().
        :return: The result of the cf.run() call.
        """
        chosen_agents = agents if agents is not None else self.agents
        return cf.run(
            objective,
            agents=chosen_agents,
            completion_agents=completion_agents,
            instructions=instructions,
            context=context or {},
            result_type=result_type,
            **kwargs
        )

    def chain_runs(self, run_specs: List[Dict[str, Any]]) -> List[Any]:
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

            # Execute the run
            result = self.run(
                objective=spec["objective"],
                agents=spec.get("agents"),
                completion_agents=spec.get("completion_agents"),
                instructions=spec.get("instructions", ""),
                context=spec.get("context"),
                result_type=spec.get("result_type", str),
                **{k: v for k, v in spec.items() if k not in ["objective", "agents", "completion_agents", "instructions", "context", "result_type"]}
            )
            results.append(result)
        return results

def run_agents(objective: str, **kwargs):
    runner = AgentRunner()
    return runner.run(objective, **kwargs)