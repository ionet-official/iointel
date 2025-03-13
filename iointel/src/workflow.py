import asyncio
import json
from typing import Dict, List, Optional, Any
import uuid

from .chainables import CHAINABLE_METHODS
from .task import CUSTOM_WORKFLOW_REGISTRY, Task

from .agent_methods.data_models.datamodels import (
    ReasoningStep,
    SummaryResult,
    TranslationResult,
    ViolationActivation,
    ModerationException,
)
from .agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
from .agent_methods.agents.agents_factory import get_agent

from iointel.client.client import (
    moderation_task,
    run_reasoning_task,
    sentiment_analysis,
    extract_entities,
    translate_text_task,
    summarize_task,
    classify_text,
    custom_workflow,
)

from pydantic import Field
from typing import Annotated
import marvin

def run_agents(objective: str, **kwargs):
    """
    A wrapper to run agent workflows synchronously.
    """
    runner = Task()
    return runner.run(objective, **kwargs)


async def run_agents_async(objective: str, **kwargs):
    """
    A wrapper to run agent workflows asynchronously.
    """
    runner = Task()
    return await runner.a_run(objective, **kwargs)


class Workflow:
    """
    A class to manage a list of tasks and run them sequentially.
    Each task is a dictionary with a "type" key that determines the type of task to run.

    # Usage example:
    my_agent = Agent(name="my_agent", instructions="Some instructions", model_provider="default")
    my_agent.run()
    reasoning_agent = Agent(name="my_agent", instructions="Some instructions")
    # Build a chain of tasks using the Tasks class

    workflow = Workflow()
    (workflow(text="Breaking news: team wins the championship!")
        .classify(["politics", "sports"], agents=[my_agent])
        .summarize_text( max_words=50, agents=[reasoning_agent]))
        .sentiment() ...

    # Run all tasks and get results
    results = workflow.run_tasks()
    print(results)


    For custom tasks:

    # Suppose you have an Agent instance:
    my_agent = Agent(name="MyAgent", instructions="Some instructions", model_provider="default")

    # Create a Workflow instance and define a custom workflow step:
    workflow = Workflow()

    (workflow("My text to process")
        .custom(
            name="do-fancy-thing",
            objective="Perform a fancy custom step on the text",
            agents=[my_agent],
            instructions="Analyze the text in a fancy custom way",
            custom_key="some_extra_value",
        )
        .council()  # chaining built-in method 'council' afterwards
        ...
       )

    results = workflow.run_tasks()
    print(results)
    """

    def __init__(
        self,
        text: str = "",
        client_mode: bool = True,
        agents: Optional[List[Any]] = None,
    ):
        self.tasks: List[dict] = []
        self.text = text
        self.client_mode = client_mode
        self.agents = agents

    def __call__(
        self, text: str, client_mode: bool = True, agents: Optional[List[Any]] = None
    ):
        self.text = text
        self.client_mode = client_mode
        self.agents = agents
        return self

    def run_tasks(self, conversation_id: Optional[str] = None, **kwargs):
        return asyncio.run(self.run_tasks_async(conversation_id, **kwargs))

    async def run_tasks_async(self, conversation_id: Optional[str] = None, **kwargs):
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        results_dict = {}

        # Implement logic to call the appropriate underlying flow based on the task type.
        # For each task in self.tasks, you call run_agents() similarly to how your flows do:
        for t in self.tasks:
            task_type = t["type"]
            agents_for_task = t.get("agents") or self.agents
            result_key = t.get("name", task_type)

            if task_type == "solve_with_reasoning":
                if self.client_mode:
                    # Use client function if you have something like `run_reasoning_task`.
                    # If not, create a similar client function or skip.
                    result = run_reasoning_task(self.text)
                    results_dict[result_key] = result
                else:
                    # logic from solve_with_reasoning flow
                    while True:
                        response: ReasoningStep = await run_agents_async(
                            objective=f"""
                            Carefully read the `goal` and analyze the problem.
                            Produce a single step of reasoning that advances you closer to a solution.

                            The goal: {self.text}

                            Here are additional instructions:
                            {REASONING_INSTRUCTIONS}
                            """,
                            result_type=ReasoningStep,
                            agents=agents_for_task or [get_agent("reasoning_agent")],
                        )
                        if response.found_validated_solution:
                            if run_agents(
                                    f"""
                                Check your solution to be absolutely sure that it is correct 
                                and meets all requirements of the goal. Return True if it does.

                                The goal: {self.text}
                                """,
                                    result_type=bool,
                                    agents=agents_for_task or [get_agent("reasoning_agent")],
                            ):
                                break
                    final = await run_agents_async(
                        objective=self.text,
                        agents=agents_for_task or [get_agent("reasoning_agent")],
                    )
                    results_dict[result_key] = final

            elif task_type == "summarize_text":
                if self.client_mode:
                    # Use client function
                    # If your `summarize_task` signature differs, adapt accordingly.
                    max_words = t["max_words"]
                    result = summarize_task(text=self.text, max_words=max_words)
                    results_dict[result_key] = result
                else:
                    summary = await run_agents_async(
                        f"Summarize the given text in no more than {t['max_words']} words and list key points. "
                        f"Here's the text: {self.text}",
                        result_type=SummaryResult,
                        agents=agents_for_task or [get_agent("summary_agent")],
                    )

                    results_dict[result_key] = summary

            elif task_type == "sentiment":
                if self.client_mode:
                    # Use client function
                    sentiment_val = sentiment_analysis(text=self.text)
                    results_dict[result_key] = sentiment_val
                else:
                    sentiment_val = await run_agents_async(
                        "Classify the sentiment of the text as a value between 0 and 1. "
                        f"Here's the text: {self.text}",
                        agents=agents_for_task or [get_agent("sentiment_analysis_agent")],
                        result_type=Annotated[float, Field(ge=0, le=1)],
                    )

                    results_dict[result_key] = sentiment_val

            elif task_type == "extract_categorized_entities":
                if self.client_mode:
                    # Use client function, e.g. `extract_entities(text=self.text)`
                    # If that function returns the same structure, great; otherwise adapt.
                    extracted = extract_entities(text=self.text)
                    results_dict[result_key] = extracted

                else:
                    extracted = await run_agents_async(
                        f"""
                        Extract named entities from the text and categorize them.

                        Return a dictionary with the following keys:
                        - 'persons': List of person names
                        - 'organizations': List of organization names
                        - 'locations': List of location names
                        - 'dates': List of date references
                        - 'events': List of event names
                        Only include keys if entities of that type are found in the text.

                        Here's the text: {self.text}
                        """,
                        agents=agents_for_task or [get_agent("extractor")],
                        result_type=Dict[str, List[str]],
                    )

                    results_dict[result_key] = extracted

            elif task_type == "translate_text":
                target_lang = t["target_language"]
                if self.client_mode:
                    # Use client function
                    translated = translate_text_task(
                        text=self.text, target_language=target_lang
                    )
                    results_dict[result_key] = translated
                else:
                    translated = await run_agents_async(
                        f"""
                        Translate the given text to {target_lang}.
                        Here's the text: {self.text}.
                        """,
                        result_type=TranslationResult,
                        agents=agents_for_task or [get_agent("translation_agent")],
                    )
                    results_dict[result_key] = translated.translated

            elif task_type == "classify":
                if self.client_mode:
                    # Use client function
                    classification = classify_text(
                        text=self.text, classify_by=t["classify_by"]
                    )
                    results_dict[result_key] = classification
                else:
                    classification = await run_agents_async(
                        f"""
                        Classify the news headline into the most appropriate category.

                        Here's the headline: {self.text}.
                        """,
                        agents=agents_for_task or [get_agent("classification_agent")],
                        result_type=t["classify_by"],
                    )
                    results_dict[result_key] = classification

            elif task_type == "moderation":
                def raise_moderation_exeption(result: dict, threshold: float):
                    if result.get("extreme_profanity", 0) > t["threshold"]:
                        raise ModerationException("Extreme profanity detected")
                    elif result.get("sexually_explicit", 0) > t["threshold"]:
                        raise ModerationException(
                            "Sexually explicit content detected"
                        )
                    elif result.get("hate_speech", 0) > t["threshold"]:
                        raise ModerationException("Hate speech detected")
                    elif result.get("harassment", 0) > t["threshold"]:
                        raise ModerationException("Harassment detected")
                    elif result.get("self_harm", 0) > t["threshold"]:
                        raise ModerationException("Self harm detected")
                    elif result.get("dangerous_content", 0) > t["threshold"]:
                        raise ModerationException("Dangerous content detected")

                if self.client_mode:
                    # Use client function
                    result = moderation_task(
                        text=self.text, threshold=t["threshold"]
                    )
                else:
                    result: ViolationActivation = await run_agents_async(
                        f"""
                        Check the text for violations and return the activation levels.

                        Here's the text: {self.text}.
                        """,
                        agents=agents_for_task or [get_agent("moderation_agent")],
                        result_type=ViolationActivation,
                    )
                raise_moderation_exeption(result, t["threshold"])
                results_dict[result_key] = result

            # Now handle "custom" tasks
            elif task_type == "custom":
                name = t["name"]
                if name in CUSTOM_WORKFLOW_REGISTRY:
                    # A registered custom function
                    custom_fn = CUSTOM_WORKFLOW_REGISTRY[name]
                    result = custom_fn(t, run_agents, self.text)
                    results_dict[result_key] = result
                else:
                    if self.client_mode:
                        # Call your client function for custom workflows, e.g.:
                        result = custom_workflow(
                            name=t["name"],
                            objective=t["objective"],
                            instructions=t.get("instructions", ""),
                            agents=agents_for_task or [get_agent("default_agent")],
                            context={**t.get("kwargs", {}), "text": self.text},
                        )
                        results_dict[result_key] = result
                    else:
                        # fallback logic if no specific function is found
                        context_formatted = json.dumps({"text": self.text, **t.get("kwargs", {})}, indent=4)

                        result = await run_agents_async(
                            f"""
                            Task objective: {t["objective"]}
                            Task instructions: {t.get("instructions", "")}
                            Additional task context: {context_formatted}
                            """,
                            agents=agents_for_task or [get_agent("default_agent")],
                            result_type=str,
                        )
                        results_dict[result_key] = result

        # Clear tasks after running
        self.tasks.clear()
        return {"conversation_id": conversation_id, "results": results_dict}


# Add chainable methods to Tasks class
for method_name, func in CHAINABLE_METHODS.items():
    setattr(Workflow, method_name, func)
