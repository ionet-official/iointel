from .workflow import Workflow

from typing import Dict, List, Optional
import uuid
import asyncio
import controlflow as cf
from .chainables import CHAINABLE_METHODS
from .workflow import CUSTOM_WORKFLOW_REGISTRY, Workflow


def run_agents(
    objective: str,
    run_async: Optional[bool] = False,
    conversation_id: Optional[str] = None,
    **kwargs
    ):
    """
    
    A wrapper to either run agent workflows synchronously or asynchronously.
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    with cf.Flow(thread_id=conversation_id):
        # Now any calls to cf.run, cf.run_async, or your Workflow here
        # will be recorded under this thread's history.
        runner = Workflow()
        if run_async:
            return runner.a_run(objective, **kwargs)  # returns a coroutine
        else:
            return runner.run(objective, **kwargs)

    return {
        "conversation_id": conversation_id,
        "result": result
    }    



class Tasks:
    """
    A class to manage a list of tasks and run them sequentially.
    Each task is a dictionary with a "type" key that determines the type of task to run.
    
    # Usage example:
    my_agent = Agent(name="my_agent", instructions="Some instructions", model_provider="default")
    my_agent.run()
    reasoning_agent = Agent(name="my_agent", instructions="Some instructions")
    # Build a chain of tasks using the Tasks class

    tasks = Tasks()
    (tasks(text="Breaking news: team wins the championship!")
        .classify(["politics", "sports"], agents=[my_agent])
        .summarize_text( max_words=50, agents=[reasoning_agent]))
        .sentiment() ...

    # Run all tasks and get results
    results = tasks.run_tasks()
    print(results)


    For custom tasks:

    # Suppose you have an Agent instance:
    my_agent = Agent(name="MyAgent", instructions="Some instructions", model_provider="default")

    # Create a Tasks instance and define a custom workflow step:
    tasks = Tasks()

    (tasks("My text to process")
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

    results = tasks.run_tasks()
    print(results)
    """

    def __init__(self, text: str = "", client_mode: bool = True, run_async: Optional[bool] = False):
        self.tasks = []
        self.text = text
        self.client_mode=client_mode
        self.run_async = run_async

    def __call__(self, text: str, client_mode: bool = True, run_async: Optional[bool] = False):
        self.text = text
        self.client_mode = client_mode
        self.run_async = run_async
        return self

    def run_tasks(self, conversation_id: Optional[str] = None, **kwargs):


        from controlflow.tasks.validators import between
        from .agent_methods.data_models.datamodels import (AgentParams, ReasoningStep, SummaryResult, 
                                                     TranslationResult, ViolationActivation,
                                                     ModerationException)
        from .agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
        from .code_parsers.pycode_parser import PythonModule
        from .code_parsers.jscode_parser import JavaScriptModule
        from framework.apis.client.client import (moderation_task, run_council_task, run_reasoning_task, 
                                          sentiment_analysis, extract_entities, translate_text_task, 
                                          summarize_task, schedule_task, classify_text, custom_workflow)

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        results_dict = {}

        # Implement logic to call the appropriate underlying flow based on the task type.
        # For each task in self.tasks, you call run_agents() similarly to how your flows do:
        for t in self.tasks:

            task_type = t["type"]
            agents_for_task = t.get("agents", None)
            result_key = t.get("name", task_type)

            if task_type == "schedule_reminder":
                if self.client_mode:
                    # Use client function
                    # If your schedule_task function needs more params (e.g. delay),
                    # you can adapt this call accordingly.
                    result = schedule_task(command=self.text)

                else:
                    if self.run_async:
                        # matches schedule_reminder_flow(command: str, delay: int)
                        async_response = run_agents(
                            objective="Schedule a reminder",
                            instructions="""
                                Schedule a reminder and use the tool to track the time for the reminder.
                            """,
                            agents=agents_for_task,
                            context={"command": self.text},
                            result_type=str,
                            run_async=True,
                            conversation_id=conversation_id
                        )
                        result = asyncio.run(async_response)["result"]
                    else:
                        # matches schedule_reminder_flow(command: str, delay: int)
                        sync_response = run_agents(
                            objective="Schedule a reminder",
                            instructions="""
                                Schedule a reminder and use the tool to track the time for the reminder.
                            """,
                            agents=agents_for_task,
                            context={"command": self.text},
                            result_type=str,
                            conversation_id=conversation_id
                        )
                        result = sync_response["result"]

                results_dict[result_key] = result

            elif task_type == "council":
                # these agents are hardcoded for now, because its a advanced workflow
                from .agent_methods.agents.agents_factory import get_agent
                from .agent_methods.agents.agents_factory import create_agent
                leader = get_agent("leader")
                council_member1 = get_agent("council_member1")
                council_member2 = get_agent("council_member2")
                council_member3 = get_agent("council_member3")
                agent_maker = get_agent("agent_maker")

                if self.client_mode:
                    # Use client function, passing self.text
                    result = run_council_task(task=self.text)
                    results_dict[result_key] = result
                else:
                    if self.run_async:
                        deliberate_coro = run_agents(
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
                            result_type=str,
                            run_async=True,
                            conversation_id=conversation_id
                        )
                        #codes = run_agents(   #WIP
                        #    "Write code for the task",
                        #    agents=[coder],
                        #    instructions="""
                        #        Provide Python or javascript code to accomplish the task depending on the user's choice.
                        #        Returns code as a pydantic model.
                        #    """,
                        #    context={"deliberation": deliberate},
                        #    result_type=PythonModule | JavaScriptModule,
                        #    run_async=True
                        #)
                        deliberate_dict = asyncio.run(deliberate_coro)
                        deliberation_result = deliberate_dict["result"]
                        
                        custom_agent_params_coro = run_agents(
                            "Create a agent to complete the task",
                            agents=[agent_maker],
                            context={"deliberation": deliberation_result},
                            result_type=AgentParams,
                            run_async=True,
                            conversation_id=conversation_id
                        )

                        custom_agent_params_dict = asyncio.run(custom_agent_params_coro)
                        agent_params_obj = custom_agent_params_dict["result"]

                        final_result = run_agents(
                        "Execute the agent to complete the task",
                        agents=[create_agent(agent_params_obj)],
                        result_type=str,
                        run_async=True,
                        conversation_id=conversation_id
                        )
                        
                        results_dict[result_key] = asyncio.run(final_result)["result"]

                    else:
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
                            result_type=str,
                            conversation_id=conversation_id
                        )
                        #codes = run_agents(   #WIP
                        #    "Write code for the task",
                        #    agents=[coder],
                        #    instructions="""
                        #        Provide Python or javascript code to accomplish the task depending on the user's choice.
                        #        Returns code as a pydantic model.
                        #    """,
                        #    context={"deliberation": deliberate},
                        #    result_type=PythonModule | JavaScriptModule
                        #)

                        custom_agent_params = run_agents(
                            "Create a agent to complete the task",
                            agents=[agent_maker],
                            context={"deliberation": deliberate["result"]},
                            result_type=AgentParams,
                            conversation_id=conversation_id
                        )

                        final_result = run_agents(
                            "Execute the agent to complete the task",
                            agents=[create_agent(custom_agent_params["result"])],
                            result_type=str,
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = final_result["result"]

            elif task_type == "solve_with_reasoning":
                if self.client_mode:
                    # Use client function if you have something like `run_reasoning_task`.
                    # If not, create a similar client function or skip.
                    result = run_reasoning_task(self.text)
                    results_dict[result_key] = result
                else:
                # logic from solve_with_reasoning flow
                    if self.run_async:
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
                                run_async=True,
                                conversation_id=conversation_id
                            )
                            res = asyncio.run(response)["result"]
                            if res.found_validated_solution:
                                run_coro = run_agents(
                                    """
                                    Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                                    """,
                                    result_type=bool,
                                    context=dict(goal=self.text),
                                    run_async=True,
                                    conversation_id=conversation_id
                                )
                                if asyncio.run(run_coro)["result"]:
                                    break
                        final = run_agents(objective=self.text, agents=agents_for_task, run_async=True, conversation_id=conversation_id)
                        results_dict[result_key] = asyncio.run(final)["result"]
                    else:
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
                                conversation_id=conversation_id
                            )
                            if response.found_validated_solution:
                                if run_agents(
                                    """
                                    Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                                    """,
                                    result_type=bool,
                                    context=dict(goal=self.text),
                                    conversation_id=conversation_id
                                ):
                                    break
                        final = run_agents(objective=self.text, agents=agents_for_task, conversation_id=conversation_id)
                        results_dict[result_key] = final["result"]

            elif task_type == "summarize_text":
                if self.client_mode:
                    # Use client function
                    # If your `summarize_task` signature differs, adapt accordingly.
                    max_words = t["max_words"]
                    result = summarize_task(text=self.text, max_words=max_words)
                    results_dict[result_key] = result
                else:
                    if self.run_async:
                        summary_coro = run_agents(
                            f"Summarize the given text in no more than {t['max_words']} words and list key points",
                            result_type=SummaryResult,
                            context={"text": self.text},
                            agents=agents_for_task,
                            run_async=True,
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = asyncio.run(summary_coro)["result"]
                        
                    else:
                        summary = run_agents(
                            f"Summarize the given text in no more than {t['max_words']} words and list key points",
                            result_type=SummaryResult,
                            context={"text": self.text},
                            agents=agents_for_task,
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = summary["result"]

            elif task_type == "sentiment":
                if self.client_mode:
                    # Use client function
                    sentiment_val = sentiment_analysis(text=self.text)
                    results_dict[result_key] = sentiment_val
                else:
                    if self.run_async:
                        sentiment_val_coro = run_agents(
                            "Classify the sentiment of the text as a value between 0 and 1",
                            agents=agents_for_task,
                            result_type=float,
                            result_validator=between(0, 1),
                            context={"text": self.text},
                            run_async=True,
                            conversation_id=conversation_id
                        )
                        results_dict[result_key] = asyncio.run(sentiment_val_coro)["result"]

                    else:
                        sentiment_val = run_agents(
                            "Classify the sentiment of the text as a value between 0 and 1",
                            agents=agents_for_task,
                            result_type=float,
                            result_validator=between(0, 1),
                            context={"text": self.text},
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = sentiment_val

            elif task_type == "extract_categorized_entities":
                if self.client_mode:
                    # Use client function, e.g. `extract_entities(text=self.text)`
                    # If that function returns the same structure, great; otherwise adapt.
                    extracted = extract_entities(text=self.text)
                    results_dict[result_key] = extracted

                else:
                    if self.run_async:
                        extracted_coro = run_agents(
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
                            run_async=True,
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = asyncio.run(extracted_coro)["result"]
                    else:
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
                            conversation_id=conversation_id
                        )

                    results_dict[result_key] = extracted["result"]

            elif task_type == "translate_text":
                target_lang = t["target_language"]
                if self.client_mode:
                    # Use client function
                    translated = translate_text_task(text=self.text, target_language=target_lang)
                    results_dict[result_key] = translated
                else:
                    if self.run_async:
                        translated = run_agents(
                            f"Translate the given text to {target_lang}",
                            result_type=TranslationResult,
                            context={"text": self.text, "target_language": target_lang},
                            agents=agents_for_task,
                            run_async=True,
                            conversation_id=conversation_id
                        )

                        results_dict[result_key] = asyncio.run(translated)["result"]
                    else:
                        translated = run_agents(
                            f"Translate the given text to {target_lang}",
                            result_type=TranslationResult,
                            context={"text": self.text, "target_language": target_lang},
                            agents=agents_for_task,
                            conversation_id=conversation_id
                        )
                        results_dict[result_key] = translated.translated

            elif task_type == "classify":
                if self.client_mode:
                    # Use client function
                    classification = classify_text(text=self.text, classify_by=t["classify_by"])
                    results_dict[result_key] = classification
                else:
                    if self.run_async:
                        classification = run_agents(
                            "Classify the news headline into the most appropriate category",
                            agents=agents_for_task,
                            result_type=t["classify_by"],
                            context={"headline": self.text},
                            run_async=True,
                            conversation_id=conversation_id
                        )
                        results_dict[result_key] = asyncio.run(classification)["result"]
                    else:
                        classification = run_agents(
                            "Classify the news headline into the most appropriate category",
                            agents=agents_for_task,
                            result_type=t["classify_by"],
                            context={"headline": self.text},
                            conversation_id=conversation_id
                        )
                        results_dict[result_key] = classification["result"]

            elif task_type == "moderation":
                if self.client_mode:
                    # Use client function
                    result = moderation_task(text=self.text, threshold=t["threshold"])
                    # If moderation_task raises exceptions or returns codes, adapt as needed:
                    if result.get("extreme_profanity", 0) > t["threshold"]:
                        raise ModerationException("Extreme profanity detected")
                    elif result.get("sexually_explicit", 0) > t["threshold"]:
                        raise ModerationException("Sexually explicit content detected")
                    elif result.get("hate_speech", 0) > t["threshold"]:
                        raise ModerationException("Hate speech detected")
                    elif result.get("harassment", 0) > t["threshold"]:
                        raise ModerationException("Harassment detected")
                    elif result.get("self_harm", 0) > t["threshold"]:
                        raise ModerationException("Self harm detected")
                    elif result.get("dangerous_content", 0) > t["threshold"]:
                        raise ModerationException("Dangerous content detected")
                    results_dict[result_key] = result
                else:
                    if self.run_async:
                        result_coro: ViolationActivation = run_agents(
                            "Check the text for violations and return the activation levels",
                            agents=agents_for_task,
                            result_type=ViolationActivation,
                            context={"text": self.text},
                            run_async=True,
                            conversation_id=conversation_id
                        )
                        result = asyncio.run(result_coro)["result"]
                    else:
                        result: ViolationActivation = run_agents(
                            "Check the text for violations and return the activation levels",
                            agents=agents_for_task,
                            result_type=ViolationActivation,
                            context={"text": self.text},
                            conversation_id=conversation_id
                        )["result"]
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
                            agents=agents_for_task,
                            context={**t.get("kwargs", {}), "text": self.text}
                        )
                        results_dict[result_key] = result
                    else:
                        if self.run_async:
                            result = run_agents(
                                objective=t["objective"],
                                instructions=t.get("instructions", ""),
                                agents=agents_for_task,
                                context={"text": self.text, **t.get("kwargs", {})},
                                result_type=str,
                                run_async=True,
                                conversation_id=conversation_id
                            )
                            results_dict[result_key] = asyncio.run(result)["result"]
                        else:
                            # fallback logic if no specific function is found
                            result = run_agents(
                                objective=t["objective"],
                                instructions=t.get("instructions", ""),
                                agents=agents_for_task,
                                context={"text": self.text, **t.get("kwargs", {})},
                                result_type=str,
                                conversation_id=conversation_id
                            )
                            results_dict[result_key] = result

        # Clear tasks after running
        self.tasks.clear()
        return {
            "conversation_id": conversation_id,
            "results": results_dict
        }

# Add chainable methods to Tasks class
for method_name, func in CHAINABLE_METHODS.items():
    setattr(Tasks, method_name, func)
