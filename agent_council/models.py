from langchain_openai import ChatOpenAI
import controlflow as cf
import os
from typing import List, Dict, Any


class Agent:

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
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        **model_kwargs
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param tools: A list of tool instances.
        :param model_name: The name of the model (e.g. "gpt-4", "gpt-3.5-turbo").
        :param api_key: The API key for the chosen model. Defaults to OPENAI_API_KEY from environment.
        :param model_kwargs: Additional keyword arguments passed to the model constructor.
        """

        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Since we are using the openAI api chat by default, we initialize it here.
        
        self.model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                **model_kwargs
            )


        # Create the cf.Agent using the configured model, instructions, and tools
        self.agent = cf.Agent(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model
        )

    def run(self, prompt: str):
        """
        Runs the agent on a given prompt. This uses cf.Agent's run method to produce a response.
        """
        return self.agent.run(prompt)

    def set_instructions(self, new_instructions: str):
        """
        Update the agent's instructions at runtime if needed.
        """
        self.instructions = new_instructions
        self.agent.instructions = new_instructions

    def add_tool(self, tool):
        """
        Add a new tool to the agent's toolkit dynamically.
        """
        self.tools.append(tool)
        self.agent.tools = self.tools


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