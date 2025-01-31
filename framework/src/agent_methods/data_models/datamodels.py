import sys
from pydantic import BaseModel, Field
from typing import List, Annotated, Optional,Union, Callable
from datetime import datetime
import controlflow
from controlflow.memory.memory import Memory
from controlflow.memory.async_memory import AsyncMemory
if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

class PersonaConfig(BaseModel):
    """
    A configuration object that describes an agent's persona or character.
    """
    name: Optional[str] = Field(
        None,
        description="If the persona has a specific name or nickname."
    )
    age: Optional[int] = Field(
        None,
        description="Approximate age of the persona (if relevant).",
        ge=1
    )
    role: Optional[str] = Field(
        None,
        description="General role or type, e.g. 'a brave knight', 'a friendly teacher', etc."
    )
    style: Optional[str] = Field(
        None,
        description="A short description of the agent's style or demeanor (e.g., 'formal and polite')."
    )
    domain_knowledge: List[str] = Field(
        default_factory=list,
        description="List of domains or special areas of expertise the agent has."
    )
    quirks: Optional[str] = Field(
        None,
        description="Any unique quirks or mannerisms, e.g. 'likes using puns' or 'always references coffee.'"
    )
    bio: Optional[str] = Field(
        None,
        description="A short biography or personal background for the persona."
    )
    lore: Optional[str] = Field(
        None,
        description="In-universe lore or backstory, e.g. 'grew up in a small village with magical powers.'"
    )
    personality: Optional[str] = Field(
        None,
        description="A more direct statement of the persona's emotional or psychological traits."
    )
    conversation_style: Optional[str] = Field(
        None,
        description="How the character speaks in conversation, e.g., 'often uses slang' or 'very verbose and flowery.'"
    )
    description: Optional[str] = Field(
        None,
        description="A general descriptive text, e.g., 'A tall, lean figure wearing a cloak, with a stern demeanor.'"
    )

    friendliness: Optional[float] = Field(
        None,
        description="How friendly the agent is, from 0 (hostile) to 1 (friendly).",
        ge=0,
        le=1
    )
    creativity: Optional[float] = Field(
        None,
        description="How creative the agent is, from 0 (very logical) to 1 (very creative).",
        ge=0,
        le=1
    )
    curiosity: Optional[float] = Field(
        None,
        description="How curious the agent is, from 0 (disinterested) to 1 (very curious).",        
        ge=0,
        le=1
    )
    empathy: Optional[float] = Field(
        None,
        description="How empathetic the agent is, from 0 (cold) to 1 (very empathetic).",
        ge=0,
        le=1
    )
    humor: Optional[float] = Field(
        None,
        description="How humorous the agent is, from 0 (serious) to 1 (very humorous).",
        ge=0,
        le=1
    )
    formality: Optional[float] = Field(
        None,
        description="How formal the agent is, from 0 (very casual) to 1 (very formal).",
        ge=0,
        le=1
    )
    emotional_stability: Optional[float] = Field(
        None,
        description="How emotionally stable the agent is, from 0 (very emotional) to 1 (very stable).",
        ge=0,
        le=1
    )

    def to_system_instructions(self) -> str:
        """
        Combine fields into a single string that can be appended to the system instructions.
        Each field is optional; only non-empty fields get appended.
        """
        lines = []

        # 1. Possibly greet with a name or reference it
        if self.name:
            lines.append(f"Your name is {self.name}.")

        # 2. Age or approximate range
        if self.age is not None:
            lines.append(f"You are {self.age} years old (approximately).")

        # 3. High-level role or type
        if self.role:
            lines.append(f"You are {self.role}.")

        # 4. Style or demeanor
        if self.style:
            lines.append(f"Your style or demeanor is: {self.style}.")

        # 5. Domain knowledge
        if self.domain_knowledge:
            knowledge_str = ", ".join(self.domain_knowledge)
            lines.append(f"You have expertise or knowledge in: {knowledge_str}.")

        # 6. Quirks
        if self.quirks:
            lines.append(f"You have the following quirks: {self.quirks}.")

        # 7. Bio
        if self.bio:
            lines.append(f"Personal background: {self.bio}.")

        # 8. Lore
        if self.lore:
            lines.append(f"Additional lore/backstory: {self.lore}.")

        # 9. Personality
        if self.personality:
            lines.append(f"Your personality traits: {self.personality}.")

        # 10. Conversation style
        if self.conversation_style:
            lines.append(f"In conversation, you speak in this style: {self.conversation_style}.")

        # 11. General description
        if self.description:
            lines.append(f"General description: {self.description}.")
        
        # 12. Personality traits
        if self.friendliness is not None:
            lines.append(f"Your overall Friendliness from 0 to 1 is: {self.friendliness}")

        if self.creativity is not None:
            lines.append(f"Your overall Creativity from 0 to 1 is: {self.creativity}")

        if self.curiosity is not None:
            lines.append(f"Your overall Curiosity from 0 to 1 is: {self.curiosity}")

        if self.empathy is not None:
            lines.append(f"Your overall Empathy from 0 to 1 is: {self.empathy}")

        if self.humor is not None:
            lines.append(f"Your overall Humor from 0 to 1 is: {self.humor}")

        if self.formality is not None:
            lines.append(f"Your overall Formality from 0 to 1 is: {self.formality}")

        if self.emotional_stability is not None:
            lines.append(f"Your overall Emotional stability from 0 to 1 is: {self.emotional_stability}")

        # Return them joined by newlines, or any separator you prefer
        return "\n".join(lines)

##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str
    description: Optional[str] = None
    persona: Optional[PersonaConfig] = None
    model: Optional[Callable] = None
    tools: Optional[List[str]] | Optional[List[Callable]] = Field(default_factory=list)
    llm_rules: Optional[controlflow.llm.rules.LLMRules] = None
    interactive: Optional[bool] = False
    memories: Optional[list[Memory]] | Optional[list[AsyncMemory]] = Field(default_factory=list)

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
    key_points: List[str]

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


class WorkflowStep(BaseModel):
    type: str
    name: Optional[str] = None
    objective: Optional[str] = None
    instructions: Optional[str] = None
    text: Optional[str] = None  # for this step’s prompt
    context: Optional[dict] = None
    target_language: Optional[str] = None

class WorkflowDefinition(BaseModel):
    """
    The top-level structure of the YAML.
    name: A human-readable name for the workflow
    agents: The agent definitions
    workflow: The list of steps
    """
    name: str
    text: Optional[str] = None  # This holds the main text/prompt
    agents: Optional[List[AgentParams]] = None
    workflow: List[WorkflowStep] = Field(default_factory=list)



### logging handlers

class BaseEventModel(BaseModel):
    """
    A base model to capture common fields or structure for all events.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentMessageEvent(BaseEventModel):
    event_type: str = "agent_message"
    agent_name: str
    content: str

class UserMessageEvent(BaseEventModel):
    event_type: str = "user_message"
    content: str

class OrchestratorMessageEvent(BaseEventModel):
    event_type: str = "orchestrator_message"
    content: str

class ToolCallEvent(BaseEventModel):
    event_type: str = "tool_call"
    tool_name: str

class ToolResultEvent(BaseEventModel):
    event_type: str = "tool_result"
    tool_name: str
    result: str

class OrchestratorStartEvent(BaseEventModel):
    event_type: str = "orchestrator_start"

class OrchestratorEndEvent(BaseEventModel):
    event_type: str = "orchestrator_end"

class AgentMessageDeltaEvent(BaseEventModel):
    event_type: str = "agent_message_delta"
    delta: str

class OrchestratorErrorEvent(BaseEventModel):
    event_type: str = "orchestrator_error"
    error: str

class EndTurnEvent(BaseEventModel):
    event_type: str = "end_turn"

class CatchallEvent(BaseEventModel):
    event_type: str = "catch-all"
    details: dict = {}

# Union of all event models
EventModelUnion = Union[
    AgentMessageEvent,
    UserMessageEvent,
    OrchestratorMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    OrchestratorStartEvent,
    OrchestratorEndEvent,
    AgentMessageDeltaEvent,
    OrchestratorErrorEvent,
    EndTurnEvent,
    CatchallEvent
]

class EventsLog(BaseModel):
    """
    Main aggregator for all events.
    """
    events: List[EventModelUnion] = []