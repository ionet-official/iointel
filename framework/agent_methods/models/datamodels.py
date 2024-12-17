from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated

##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str

#reasoning agent
class ReasoningStep(BaseModel):
    explanation: str = Field(
        description="""
            A brief (<5 words) description of what you intend to
            achieve in this step, to display to the user.
            """
    )
    reasoning: str = Field(
        description="A single step of reasoning, not more than 1 or 2 sentences."
    )
    found_validated_solution: bool


##summary
class SummaryResult(BaseModel):
    summary: str
    key_points: list[str]

#translation
class TranslationResult(BaseModel):
    translated: str
    target_language: str


Activation = Annotated[float, Field(ge=0, le=1)]


class ModerationException(Exception):
    """Exception raised when a message is not allowed."""

    ...


class ViolationActivation(TypedDict):
    """Violation activation."""

    extreme_profanity: Annotated[Activation, Field(description="hell / damn are fine")]
    sexually_explicit: Activation
    hate_speech: Activation
    harassment: Activation
    self_harm: Activation
    dangerous_content: Activation
