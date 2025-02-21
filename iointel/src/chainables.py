from typing import List, Optional, Dict, Any

# from .task import CUSTOM_WORKFLOW_REGISTRY
from .agent_methods.tools.tools import between
from .utilities.runners import run_agents
from .utilities.decorators import register_custom_task
from .utilities.registries import CHAINABLE_METHODS, CUSTOM_WORKFLOW_REGISTRY
from .agents import Agent

##############################################
# Example Executor Functions
##############################################


@register_custom_task("schedule_reminder")
def execute_schedule_reminder(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import schedule_task
    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return schedule_task(command=text)
    else:
        response = run_agents(
            objective="Schedule a reminder",
            instructions="Schedule a reminder and track the time.",
            agents=agents,
            context={"command": text},
            result_type=str,
        )
        return response.execute()


@register_custom_task("solve_with_reasoning")
def execute_solve_with_reasoning(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from .agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
    from .agent_methods.data_models.datamodels import ReasoningStep
    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        from ..client.client import run_reasoning_task

        return run_reasoning_task(text)
    else:
        # For example, loop until a validated solution is found.
        while True:
            response: ReasoningStep = run_agents(
                objective=REASONING_INSTRUCTIONS,
                result_type=ReasoningStep,
                agents=agents,
                context={"goal": text},
            ).execute()
            if response.found_validated_solution:
                # Optionally, double-check the solution.
                if run_agents(
                    objective="""
                            Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                            """,
                    result_type=bool,
                    context={"goal": text},
                    agents=agents,
                ).execute():
                    return response.proposed_solution


@register_custom_task("summarize_text")
def execute_summarize_text(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import summarize_task
    from .agent_methods.data_models.datamodels import SummaryResult

    max_words = task_metadata.get("max_words")
    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return summarize_task(text=text, max_words=max_words)
    else:
        summary = run_agents(
            objective=f"Summarize the given text in no more than {max_words} words and list key points",
            result_type=SummaryResult,
            context={"text": text},
            agents=agents,
        )
        return summary.execute()


@register_custom_task("sentiment")
def execute_sentiment(task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict):

    from ..client.client import sentiment_analysis

    client_mode = execution_metadata.get("client_mode", False)
    
    if client_mode:
        return sentiment_analysis(text=text)
    else:
        sentiment_val = run_agents(
            objective="Classify the sentiment of the text as a value between 0 and 1",
            agents=agents,
            result_type=float,
            result_validator=between(0, 1),
            context={"text": text},
        )
        return sentiment_val.execute()


@register_custom_task("extract_categorized_entities")
def execute_extract_entities(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import extract_entities
    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return extract_entities(text=text)
    else:
        extracted = run_agents(
            objective="""Extract named entities from the text and categorize them,
                            Return a dictionary with the following keys:
                            - 'persons': List of person names
                            - 'organizations': List of organization names
                            - 'locations': List of location names
                            - 'dates': List of date references
                            - 'events': List of event names
                            Only include keys if entities of that type are found in the text.
                            """,
            agents=agents,
            result_type=Dict[str, List[str]],
            context={"text": text},
        )
        return extracted.execute()


@register_custom_task("translate_text")
def execute_translate_text(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    target_lang = task_metadata["target_language"]
    from ..client.client import translate_text_task
    from .agent_methods.data_models.datamodels import TranslationResult

    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return translate_text_task(text=text, target_language=target_lang)
    else:
        translated = run_agents(
            objective=f"Translate the given text to {target_lang}",
            result_type=TranslationResult,
            context={"text": text, "target_language": target_lang},
            agents=agents,
        )
        result = translated.execute()
        # Assuming the model has an attribute 'translated'
        return result.translated


@register_custom_task("classify")
def execute_classify(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import classify_text

    client_mode = execution_metadata.get("client_mode", False)
    classify_by = task_metadata.get("classify_by")

    if client_mode:
        return classify_text(text=text, classify_by=classify_by)
    else:
        classification = run_agents(
            objective="Classify the text into the appropriate category",
            agents=agents,
            result_type=classify_by,
            context={"text": text},
        )
        return classification.execute()


@register_custom_task("moderation")
def execute_moderation(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
):
    from .agent_methods.data_models.datamodels import (
        ViolationActivation,
        ModerationException,
    )
    from ..client.client import moderation_task

    client_mode = execution_metadata.get("client_mode", False)
    threshold = task_metadata["threshold"]

    if client_mode:
        result = moderation_task(text=text, threshold=threshold)
        # Raise exceptions based on result thresholds if necessary.
        return result
    else:
        result: ViolationActivation = run_agents(
            objective="Check the text for violations and return activation levels",
            agents=agents,
            result_type=ViolationActivation,
            context={"text": text},
        ).execute()

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
            raise ModerationException("Dangeme profanity detected")

        return result


@register_custom_task("custom")
def execute_custom(
    task_metadata: dict, text: str, agents: List[Agent], execution_metadata: dict
    ):

    client_mode = execution_metadata.get("client_mode", False)
    name = task_metadata["name"]

    if name in CUSTOM_WORKFLOW_REGISTRY:
        custom_fn = CUSTOM_WORKFLOW_REGISTRY[name]
        result = custom_fn(task_metadata, run_agents, text)
        if hasattr(result, "execute") and callable(result.execute):
            result = result.execute()
        return result
    else:
        if client_mode:
            from ..client.client import custom_workflow

            return custom_workflow(
                name=task_metadata["name"],
                objective=task_metadata["objective"],
                agents=agents,
                context={**task_metadata.get("kwargs", {}), "text": text},
            )
        else:
            response = run_agents(
                objective=task_metadata["objective"],
                agents=agents,
                context={"text": text, **task_metadata.get("kwargs", {})},
                result_type=str,
            )
            return response.execute()


##############################################
# CHAINABLES
##############################################
def schedule_reminder(self, delay: int = 0, agents: Optional[List[Agent]] = None):
    # WIP
    self.tasks.append(
        {
            "type": "schedule_reminder",
            "command": self.text,
            "task_metadata": {"delay": delay},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def solve_with_reasoning(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
         "type": "solve_with_reasoning",
         "text": self.text,
         "agents": self.agents if agents is None else agents,
         }
    )
    return self


def summarize_text(self, max_words: int = 100, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "summarize_text",
            "text": self.text,
            "agents": self.agents if agents is None else agents,
            "task_metadata": {"max_words": max_words},
        }
    )
    return self


def sentiment(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
            {
                "type": "sentiment",
                "text": self.text,
                "agents": self.agents if agents is None else agents,
            }
        )
    return self


def extract_categorized_entities(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "extract_categorized_entities",
            "text": self.text,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def translate_text(self, target_language: str, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "translate_text",
            "text": self.text,
            "task_metadata": {"target_language": target_language},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def classify(self, classify_by: list, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "classify",
            "task_metadata": {"classify_by": classify_by},
            "text": self.text,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def moderation(self, threshold: float, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "moderation",
            "text": self.text,
            "task_metadata": {"threshold": threshold},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


# def custom(self, name: str, objective: str, agents: Optional[List[Agent]] = None, instructions: str = "", **kwargs):
def custom(
    self, name: str, objective: str, agents: Optional[List[Agent]] = None, **kwargs
):
    """
    Allows users to define a custom workflow (or step) that can be chained
    like the built-in tasks. 'name' can help identify the custom workflow
    in run_tasks().

    :param name: Unique identifier for this custom workflow step.
    :param objective: The main objective or prompt for run_agents.
    :param agents: List of agents used (if None, a default can be used).
    #:param instructions: Additional instructions for run_agents.
    :param kwargs: Additional data needed for this custom workflow.
    """
    self.tasks.append(
        {
            "type": "custom",
            "text": self.text,
            "task_metadata": {"name": name, "objective": objective, "kwargs": kwargs},
            "agents": self.agents if agents is None else agents,
            # "instructions": instructions,
        }
    )
    return self


# Dictionary mapping method names to functions
CHAINABLE_METHODS.update(
    {
        "schedule_reminder": schedule_reminder,
        "solve_with_reasoning": solve_with_reasoning,
        "summarize_text": summarize_text,
        "sentiment": sentiment,
        "extract_categorized_entities": extract_categorized_entities,
        "translate_text": translate_text,
        "classify": classify,
        "moderation": moderation,
        "custom": custom,
    }
)
