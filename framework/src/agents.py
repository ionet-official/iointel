from langchain_openai import ChatOpenAI
import controlflow as cf
from typing import List, Dict, Any
from framework.src.tasks import CHAINABLE_METHODS

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


        super().__init__(
            name=name,
            instructions=instructions,
            tools=tools or [],
            model=model,
            **model_kwargs
        )

    def run(self, prompt: str):
        return super().run(prompt)

    def set_instructions(self, new_instructions: str):
        self.instructions = new_instructions

    def add_tool(self, tool):
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
        agents: List[Agent] = None,
        completion_agents: List[Agent] = None,
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

@cf.flow
def run_agents(objective: str, **kwargs):
    runner = AgentRunner()
    return runner.run(objective, **kwargs)



class Tasks:
    """
    A class to manage a list of tasks and run them sequentially.
    Each task is a dictionary with a "type" key that determines the type of task to run.
    
    # Usage example:
    my_agent = Agent(name="my_agent", instructions="Some instructions")
    reasoning_agent = Agent(name="my_agent", instructions="Some instructions")
    # Build a chain of tasks using the Tasks class

    tasks = Tasks()
    (tasks(text="Breaking news: team wins the championship!")
        .classify(["politics", "sports"], agents=[my_agent])
        .summarize_text( max_words=50, agents=[reasoning_agent]))

    # Run all tasks and get results
    results = tasks.run_tasks()
    print(results)
    """

    def __init__(self, text: str = "", local: bool = False):
        self.tasks = []
        self.text = text
        self.local=local

    def run_tasks(self):
        results = []
        from framework.src.agents import run_agents
        from agent_methods.agents.agents import (leader, council_member1, council_member2, 
                                        council_member3, coder, agent_maker, 
                                        sentiment_analysis_agent, reminder_agent, reasoning_agent,
                                        extractor, default_agent, moderation_agent)
        from controlflow.tasks.validators import between
        from agent_methods.models.datamodels import (AgentParams, ReasoningStep, SummaryResult, 
                                                     TranslationResult, ViolationActivation,
                                                     ModerationException)
        from framework.src.agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
        from framework.src.agent_methods.tools.tools import create_agent
        from framework.src.code_parsers.pycode_parser import PythonModule
        from framework.src.code_parsers.jscode_parser import JavaScriptModule

        # Implement logic to call the appropriate underlying flow based on the task type.
        # For each task in self.tasks, you call run_agents() similarly to how your flows do:
        for t in self.tasks:

            task_type = t["type"]
            agents_for_task = t.get("agents", [])
            if task_type == "schedule_reminder":
                # matches schedule_reminder_flow(command: str, delay: int)
                result = run_agents(
                    objective="Schedule a reminder",
                    instructions="""
                        Schedule a reminder and use the tool to track the time for the reminder.
                    """,
                    agents=agents_for_task,
                    context={"command": self.text},
                    result_type=str,
                )
                results.append(result)

            elif task_type == "council":
                deliberate = run_agents(
                    "Deliberate and vote on the best way to complete the task.",
                    agents=[leader, council_member1, council_member2, council_member3],
                    completion_agents=[leader],
                    instructions="""
                        Deliberate with other council members on the best way to complete the task.
                        Allow each council member to provide input before voting.
                        Vote on the best answer.
                        Show the entire deliberation, voting process, final decision, and reasoning.
                    """,
                    context={"task": self.text},
                    result_type=str
                )
                codes = run_agents(
                    "Write code for the task",
                    agents=[coder],
                    instructions="""
                        Provide Python or javascript code to accomplish the task depending on the user's choice.
                        Returns code as a pydantic model.
                    """,
                    context={"deliberation": deliberate},
                    result_type=PythonModule | JavaScriptModule
                )

                custom_agent_params = run_agents(
                    "Create a ControlFlow agent using the provided code.",
                    agents=[agent_maker],
                    context={"code": codes},
                    result_type=AgentParams,
                )

                final_result = run_agents(
                    "Execute the agent to complete the task",
                    agents=[create_agent(custom_agent_params)],
                    result_type=str
                )
                results.append(final_result)

            elif task_type == "solve_with_reasoning":
                # logic from solve_with_reasoning flow
                while True:
                    response: ReasoningStep = run_agents(
                        objective="""
                        Carefully read the `goal` and analyze the problem.
                        Produce a single step of reasoning that advances you closer to a solution.
                        """,
                        instructions=REASONING_INSTRUCTIONS,
                        result_type=ReasoningStep,
                        agents=agents_for_task,
                        context=dict(goal=self.text),
                        model_kwargs=dict(tool_choice="required"),
                    )
                    if response.found_validated_solution:
                        if run_agents(
                            """
                            Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                            """,
                            result_type=bool,
                            context=dict(goal=self.text),
                        ):
                            break
                final = run_agents(objective=self.text, agents=agents_for_task)
                results.append(final)

            elif task_type == "summarize_text":
                summary = run_agents(
                    f"Summarize the given text in no more than {t['max_words']} words and list key points",
                    result_type=SummaryResult,
                    context={"text": self.text},
                    agents=agents_for_task,
                )
                results.append(summary)

            elif task_type == "sentiment":
                sentiment_val = run_agents(
                    "Classify the sentiment of the text as a value between 0 and 1",
                    agents=agents_for_task,
                    result_type=float,
                    result_validator=between(0, 1),
                    context={"text": self.text},
                )
                results.append(sentiment_val)

            elif task_type == "extract_categorized_entities":
                extracted = run_agents(
                    "Extract named entities from the text and categorize them",
                    instructions="""
                    Return a dictionary with the following keys:
                    - 'persons': List of person names
                    - 'organizations': List of organization names
                    - 'locations': List of location names
                    - 'dates': List of date references
                    - 'events': List of event names
                    Only include keys if entities of that type are found in the text.
                    """,
                    agents=agents_for_task,
                    result_type=Dict[str, List[str]],
                    context={"text": self.text},
                )
                results.append(extracted)

            elif task_type == "translate_text":
                translated = run_agents(
                    f"Translate the given text to {t['target_language']}",
                    result_type=TranslationResult,
                    context={"text": self.text, "target_language": t["target_language"]},
                    agents=agents_for_task,
                )
                results.append(translated)

            elif task_type == "classify":
                classification = run_agents(
                    "Classify the news headline into the most appropriate category",
                    agents=agents_for_task,
                    result_type=t["classify_by"],
                    context={"headline": self.text},
                )
                results.append(classification)

            elif task_type == "moderation":
                result: ViolationActivation = run_agents(
                    "Check the text for violations and return the activation levels",
                    agents=agents_for_task,
                    result_type=ViolationActivation,
                    context={"text": self.text},
                )
                threshold = t["threshold"]
                if result["extreme_profanity"] > threshold:
                    raise ModerationException("Extreme profanity detected")
                elif result["sexually_explicit"] > threshold:
                    raise ModerationException("Sexually explicit content detected")
                elif result["hate_speech"] > threshold:
                    raise ModerationException("Hate speech detected")
                elif result["harassment"] > threshold:
                    raise ModerationException("Harassment detected")
                elif result["self_harm"] > threshold:
                    raise ModerationException("Self harm detected")
                elif result["dangerous_content"] > threshold:
                    raise ModerationException("Dangerous content detected")
                results.append(result)

        # Clear tasks after running
        self.tasks.clear()
        return results

# Add chainable methods to Tasks class
for method_name, func in CHAINABLE_METHODS.items():
    setattr(Tasks, method_name, func)