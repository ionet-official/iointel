from pydantic import BaseModel, Field


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