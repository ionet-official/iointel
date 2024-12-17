import controlflow as cf
from controlflow.tasks.validators import between
from agent_methods.agents.agents import (leader, council_member1, council_member2, 
                                        council_member3, coder, agent_maker, 
                                        sentiment_analysis_agent, reminder_agent, reasoning_agent,
                                        extractor, default_agent, moderation_agent)

from agent_methods.models.datamodels import (AgentParams, ReasoningStep, SummaryResult, TranslationResult,
                                             ViolationActivation, ModerationException)
from agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
from agent_methods.tools.tools import create_agent
from framework.src.agents import run_agents
from framework.src.code_parsers.pycode_parser import PythonModule
from framework.src.code_parsers.jscode_parser import JavaScriptModule
from typing import Dict, List

@cf.flow
def schedule_reminder_flow(command: str, delay: int = 0) -> str:
    """
    A flow that schedules a reminder after a given delay.
    Instead of a persistent task manager, we just call this flow whenever we need.
    """
    
    reminder = run_agents(
        objective = "Schedule a reminder",
        instructions="""
            Schedule a reminder and use the tool to track the time for the reminder.
        """,
        agents=[reminder_agent],
        context={"command": command},
        result_type=str,
    )
    return reminder


@cf.flow
def council_flow(task: str) -> str: #generalized council flow
    # step 1: deliberation and voting process
    deliberate = run_agents(
        "Deliberate and vote on the best way to complete the task.",
        agents=[leader, council_member1, council_member2, council_member3],
        completion_agents=[leader],
        #turn_strategy=cf.orchestration.turn_strategies.Moderated(moderator=leader),
        instructions="""
            Deliberate with other council members on the best way to complete the task.
            Allow each council member to provide input before voting.
            Vote on the best answer.
            Show the entire deliberation, voting process, final decision, and reasoning.
        """,
        context={"task": task},
        result_type=str
    )
    print(deliberate)

    # Step 2: write code for the task
    codes = run_agents(
        "Write code for the task",
        agents=[coder],
        instructions="""
            Provide Python code to accomplish the task.
            Return code in a format that can be parsed by python (as a string).
        """,
        context={"deliberation": deliberate},
        result_type=PythonModule | JavaScriptModule
    )

    # Step 3: generate an agent to run the code
    custom_agent_params = run_agents(
        "Create a ControlFlow agent using the provided code.",
        agents=[agent_maker],
        context={"code": codes},
        result_type=AgentParams,
    )


    # step 4: run the agent
    result = run_agents(
        "Execute the agent to complete the task",
        agents=[create_agent(custom_agent_params)],
        result_type=str
    )
    return result


@cf.flow
def solve_with_reasoning(goal: str) -> str:
    while True:
        response: ReasoningStep = run_agents(
            objective="""
            Carefully read the `goal` and analyze the problem.
            
            Produce a single step of reasoning that advances you closer to a solution.
            """,
            instructions=REASONING_INSTRUCTIONS,
            result_type=ReasoningStep,
            agents=[reasoning_agent],
            context=dict(goal=goal),
            model_kwargs=dict(tool_choice="required"),
        )

        if response.found_validated_solution:
            if run_agents(
                """
                Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                """,
                result_type=bool,
                context=dict(goal=goal),
            ):
                break

    return run_agents(objective=goal, agents=[reasoning_agent])


@cf.flow
def summarize_text(text: str, max_words: int = 100) -> SummaryResult:
    return run_agents(
        f"Summarize the given text in no more than {max_words} words and list key points",
        result_type=SummaryResult,
        context={"text": text},
        agents=[reasoning_agent],
    )


@cf.flow
def sentiment(text: str) -> float:
    return run_agents(
        "Classify the sentiment of the text as a value between 0 and 1",
        agents=[sentiment_analysis_agent],
        result_type=float,
        result_validator=between(0, 1),
        context={"text": text},
    )


@cf.flow
def extract_categorized_entities(text: str) -> Dict[str, List[str]]:
    return run_agents(
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
        agents=[extractor],
        result_type=Dict[str, List[str]],
        context={"text": text},
    )

@cf.flow
def translate_text(text: str, target_language: str) -> TranslationResult:
    return run_agents(
        f"Translate the given text to {target_language}",
        result_type=TranslationResult,
        context={"text": text, "target_language": target_language},
        agents=[default_agent],
    )

@cf.flow
def classify(classify_by: List[str], to_be_classified: str) -> str:
    return run_agents(
        "Classify the news headline into the most appropriate category",
        agents=[default_agent],
        result_type=classify_by,
        context={"headline": to_be_classified},
    )

@cf.flow
def moderation(text: str, threshold: float) -> ViolationActivation:
    result: ViolationActivation = run_agents(
        "Check the text for violations and return the activation levels",
        agents=[moderation_agent],
        result_type=ViolationActivation,
        context={"text": text},
    )
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
    return result