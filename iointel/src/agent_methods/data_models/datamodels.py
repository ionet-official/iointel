import functools
import re
import sys
import warnings
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    SecretStr,
    field_serializer,
    field_validator,
)

from typing import (
    List,
    Annotated,
    Optional,
    Union,
    Callable,
    Dict,
    Any,
    Literal,
)
from pydantic_ai.models.openai import OpenAIModel
import weakref

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


from ...memory import AsyncMemory
from ...utilities.func_metadata import func_metadata, FuncMetadata
from ...utilities.exceptions import ToolError
from ...utilities.registries import TOOL_SELF_REGISTRY
import inspect


# monkey patching for OpenAIModel to return a generic schema
def patched_get_json_schema(cls, core_schema, handler):
    # Return a generic schema for the OpenAIModel.
    # Adjust this as needed for your application.
    return {"type": "object", "title": cls.__name__}


# Monkey-patch the __get_pydantic_json_schema__ on OpenAIModel.
OpenAIModel.__get_pydantic_json_schema__ = classmethod(patched_get_json_schema)


class ToolUsageResult(BaseModel):
    tool_name: str
    tool_args: dict
    tool_result: Any = None


###### persona ########
class PersonaConfig(BaseModel):
    """
    A configuration object that describes an agent's persona or character.
    """

    name: Optional[str] = Field(
        None, description="If the persona has a specific name or nickname."
    )
    age: Optional[int] = Field(
        None, description="Approximate age of the persona (if relevant).", ge=1
    )
    role: Optional[str] = Field(
        None,
        description="General role or type, e.g. 'a brave knight', 'a friendly teacher', etc.",
    )
    style: Optional[str] = Field(
        None,
        description="A short description of the agent's style or demeanor (e.g., 'formal and polite').",
    )
    domain_knowledge: List[str] = Field(
        default_factory=list,
        description="List of domains or special areas of expertise the agent has.",
    )
    quirks: Optional[str] = Field(
        None,
        description="Any unique quirks or mannerisms, e.g. 'likes using puns' or 'always references coffee.'",
    )
    bio: Optional[str] = Field(
        None, description="A short biography or personal background for the persona."
    )
    lore: Optional[str] = Field(
        None,
        description="In-universe lore or backstory, e.g. 'grew up in a small village with magical powers.'",
    )
    personality: Optional[str] = Field(
        None,
        description="A more direct statement of the persona's emotional or psychological traits.",
    )
    conversation_style: Optional[str] = Field(
        None,
        description="How the character speaks in conversation, e.g., 'often uses slang' or 'very verbose and flowery.'",
    )
    description: Optional[str] = Field(
        None,
        description="A general descriptive text, e.g., 'A tall, lean figure wearing a cloak, with a stern demeanor.'",
    )

    friendliness: Optional[Union[float, str]] = Field(
        None,
        description="How friendly the agent is, from 0 (hostile) to 1 (friendly).",
        ge=0,
        le=1,
    )
    creativity: Optional[Union[float, str]] = Field(
        None,
        description="How creative the agent is, from 0 (very logical) to 1 (very creative).",
        ge=0,
        le=1,
    )
    curiosity: Optional[Union[float, str]] = Field(
        None,
        description="How curious the agent is, from 0 (disinterested) to 1 (very curious).",
        ge=0,
        le=1,
    )
    empathy: Optional[Union[float, str]] = Field(
        None,
        description="How empathetic the agent is, from 0 (cold) to 1 (very empathetic).",
        ge=0,
        le=1,
    )
    humor: Optional[Union[float, str]] = Field(
        None,
        description="How humorous the agent is, from 0 (serious) to 1 (very humorous).",
        ge=0,
        le=1,
    )
    formality: Optional[Union[float, str]] = Field(
        None,
        description="How formal the agent is, from 0 (very casual) to 1 (very formal).",
        ge=0,
        le=1,
    )
    emotional_stability: Optional[Union[float, str]] = Field(
        None,
        description="How emotionally stable the agent is, from 0 (very emotional) to 1 (very stable).",
        ge=0,
        le=1,
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
            lines.append(
                f"In conversation, you speak in this style: {self.conversation_style}."
            )

        # 11. General description
        if self.description:
            lines.append(f"General description: {self.description}.")

        # 12. Personality traits
        if self.friendliness is not None:
            lines.append(
                f"Your overall Friendliness from 0 to 1 is: {self.friendliness}"
            )

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
            lines.append(
                f"Your overall Emotional stability from 0 to 1 is: {self.emotional_stability}"
            )

        # Return them joined by newlines, or any separator you prefer
        return "\n".join(lines)


def compute_fn_name(fn: Callable) -> str:
    # OpenAI chokes on non-conforming tool names, so scrub them
    return re.sub(r"[^a-zA-Z0-9_-]", "-", fn.__qualname__)


def check_fn_name(name: str | None) -> str | None:
    if name and not re.match(r"^[a-zA-Z0-9_-]+", name):
        warnings.warn(f"Tool name {name} is not compatible with OpenAI")
    return name


# mapping from id(instance) to instance
TOOL_SELF_INSTANCES: dict[int, BaseModel] = weakref.WeakValueDictionary()


class Tool(BaseModel):
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: dict = Field(description="JSON schema for tool parameters")
    is_async: bool = Field(description="Whether the tool is async")
    body: Optional[str] = Field(None, description="Source code of the tool function")
    # fn and fn_metadata are excluded from serialization.
    fn: Optional[Callable] = Field(default=None, exclude=True)
    fn_metadata: Optional[FuncMetadata] = Field(default=None, exclude=True)
    fn_self: Optional[tuple[str, str, int]] = Field(
        None, description="Serialised `self` if `fn` is an instance method"
    )

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _instance_tool_key(tool_self: BaseModel) -> str:
        return f"{tool_self.__class__.__module__}:{tool_self.__class__.__name__}"

    @field_validator("fn", mode="after")
    @classmethod
    def check_supported_fn(cls, value: Optional[Callable]) -> Callable:
        if not value:
            raise ValueError("Tool got empty `fn`")
        if inspect.ismethod(value):
            if not isinstance(value.__self__, BaseModel):
                raise ValueError(
                    f"When defining instance method tool, class {value.__self__.__class__} must inherit from BaseModel"
                )
            self_type = value.__self__.__class__
            self_key = cls._instance_tool_key(value.__self__)
            if old_value := TOOL_SELF_REGISTRY.get(self_key):
                if old_value != self_type:
                    raise ValueError(
                        f"Different classes with same key {self_key} detected, refusing to use tool {value}"
                    )
            else:
                TOOL_SELF_REGISTRY[self_key] = self_type
            return value
        if not (
            inspect.iscoroutinefunction(value)
            or inspect.isfunction(value)
            or isinstance(value, functools.partial)
        ):
            raise ValueError(f"Unsupported tool type: {type(value)}")
        return value

    @field_serializer("fn_self")
    def serialize_fn_self(self, _: Any):
        if self.fn_self is not None:
            return self.fn_self
        if not inspect.ismethod(self.fn) or not isinstance(self.fn.__self__, BaseModel):
            return None
        self._update_fn_self(self.fn.__self__)
        return self.fn_self

    def _update_fn_self(self, fn_self: BaseModel):
        tool_key = self._instance_tool_key(fn_self)
        assert tool_key in TOOL_SELF_REGISTRY
        TOOL_SELF_INSTANCES[id(fn_self)] = fn_self
        self.fn_self = tool_key, fn_self.model_dump_json(), id(fn_self)

    def _load_fn_self(self) -> Optional[BaseModel]:
        if not self.fn_self:
            return None
        tool_key, tool_json, tool_id = self.fn_self
        if (instance := TOOL_SELF_INSTANCES.get(tool_id)) is not None:
            return instance
        return TOOL_SELF_REGISTRY[tool_key].model_validate_json(tool_json)

    def instantiate_from_state(self, state: BaseModel) -> "Tool":
        tool_fn = getattr(state, self.fn.__name__)
        self.check_supported_fn(tool_fn)
        tool_key = self._instance_tool_key(state)
        return self.model_copy(
            update={"fn_self": (tool_key, state.model_dump_json(), id(state))}
        )

    def __call__(self, *args, **kwargs):
        if self.fn:
            return self.fn(*args, **kwargs)
        raise ValueError(f"Tool {self.name} has not been rehydrated correctly.")

    def get_wrapped_fn(self) -> callable:
        fn_self = self._load_fn_self()
        if fn_self or inspect.ismethod(self.fn):
            __tool_self = fn_self or self.fn.__self__
            __tool_fn = self.fn.__func__ if inspect.ismethod(self.fn) else self.fn
            __self = self
            annotations = dict(inspect.get_annotations(__tool_fn))
            annotations.pop("self", None)
            sig = inspect.signature(self.fn)
            if "self" in sig.parameters:
                new_args = dict(sig.parameters)
                new_args.pop("self", None)
                sig = inspect.Signature(
                    new_args.values(), return_annotation=sig.return_annotation
                )
            if self.is_async:

                @functools.wraps(__tool_fn)
                async def wrapper(*a, **kw):
                    try:
                        return await __tool_fn(__tool_self, *a, **kw)
                    finally:
                        __self._update_fn_self(__tool_self)
            else:

                @functools.wraps(__tool_fn)
                def wrapper(*a, **kw):
                    try:
                        return __tool_fn(__tool_self, *a, **kw)
                    finally:
                        __self._update_fn_self(__tool_self)

            wrapper.__annotations__ = annotations
            wrapper.__signature__ = sig
            return wrapper

        return self.fn

    @property
    def __name__(self):
        if self.fn and hasattr(self.fn, "__name__"):
            return self.fn.__name__
        return self.name  # fallback to the Tool's name

    @classmethod
    def from_function(
        cls, fn: Callable, name: Optional[str] = None, description: Optional[str] = None
    ) -> "Tool":
        if isinstance(fn, cls):
            return fn
        func_name = check_fn_name(name) or compute_fn_name(fn)
        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")
        func_doc = description or fn.__doc__ or ""
        is_async = inspect.iscoroutinefunction(fn)
        func_arg_metadata = func_metadata(fn)
        parameters = func_arg_metadata.arg_model.model_json_schema()
        try:
            body = inspect.getsource(fn)
        except Exception:
            body = None
        return cls(
            fn=fn,
            name=func_name,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            body=body,
        )

    async def run(self, arguments: dict) -> Any:
        """Run the tool with arguments."""
        try:
            return await self.fn_metadata.call_fn_with_arg_validation(
                self.fn, self.is_async, arguments
            )
        except Exception as e:
            raise ToolError(f"Error executing tool {self.name}: {e}") from e


##agent params###
class AgentParams(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    name: Optional[str] = None
    instructions: str = Field(..., description="Instructions for the agent")
    persona: Optional[PersonaConfig] = None
    model: Optional[Union[OpenAIModel, str]] = Field(
        None,
        description="Model or model name for the agent",
    )
    api_key: Optional[Union[str, SecretStr]] = Field(
        None, description="API key for the model, if required."
    )
    base_url: Optional[str] = Field(
        None, description="Base URL for the model, if required."
    )
    tools: Optional[List[str | dict | tuple[str, dict] | Tool | Callable]] = Field(
        default_factory=list
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Context to be passed to the agent.",
    )
    # memories: Optional[list[Memory]] = Field(default_factory=list)
    memory: Optional[AsyncMemory] = None

    model_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    output_type: Optional[Any] = str

    @field_serializer("output_type", when_used="json")
    def dump_output_type(self, v):
        if isinstance(v, type):
            # convert builtin and some global-accessible types to their string names
            from iointel.src.agent_methods.agents.agents_factory import create_agent

            name = v.__name__
            if v is create_agent.__globals__.get(name, __builtins__.get(name, None)):
                return name
        return v


# reasoning agent
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
    proposed_solution: str = Field(description="The proposed solution for the problem.")


class AgentSwarm(BaseModel):
    members: List[AgentParams]


##summary
class SummaryResult(BaseModel):
    summary: str
    key_points: List[str]


# translation
class TranslationResult(BaseModel):
    translated: str
    target_language: str


Activation = Annotated[float, Field(ge=0, le=1)]


class ViolationActivation(TypedDict):
    """Violation activation."""

    extreme_profanity: Annotated[Activation, Field(description="hell / damn are fine")]
    sexually_explicit: Activation
    hate_speech: Activation
    harassment: Activation
    self_harm: Activation
    dangerous_content: Activation


class ModerationException(Exception):
    """Exception raised when a message is not allowed."""

    def __init__(self, *args, violations: ViolationActivation):
        super().__init__(*args)
        self.violations = violations


##### task and workflow models ########
class BaseStage(BaseModel):
    stage_id: Optional[int] = None
    stage_name: str = ""


class SimpleStage(BaseStage):
    stage_type: Literal["simple"] = "simple"
    objective: str
    output_type: Any = None
    agents: List[Union[AgentParams, AgentSwarm]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SequentialStage(BaseStage):
    stage_type: Literal["sequential"] = "sequential"
    stages: List["Stage"] = Field(
        ..., description="List of stages to execute sequentially"
    )


class ParallelStage(BaseStage):
    stage_type: Literal["parallel"] = "parallel"
    # merge_strategy: Optional[str] = None
    stages: List["Stage"] = Field(
        ..., description="List of stages to execute in parallel"
    )


class WhileStage(BaseStage):
    stage_type: Literal["while"] = "while"
    condition: str | Callable = Field(
        ...,
        description=(
            "A condition (expressed as a string or expression) that determines whether "
            "the loop should continue. The evaluation of this condition should be handled "
            "by the executor logic."
        ),
    )
    max_iterations: Optional[int] = Field(
        100,
        description="An optional safeguard to limit the number of iterations and prevent infinite loops.",
    )
    stage: List["Stage"] = Field(..., description="The loop body")


class FallbackStage(BaseStage):
    stage_type: Literal["fallback"] = "fallback"
    primary: "Stage" = Field(..., description="The primary stage to execute")
    fallback: "Stage" = Field(
        ..., description="The fallback stage to execute if primary fails"
    )


Stage = Union[SimpleStage, ParallelStage, WhileStage, FallbackStage]


FallbackStage.model_rebuild()
ParallelStage.model_rebuild()
SequentialStage.model_rebuild()
WhileStage.model_rebuild()


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    type: str = "custom"
    # description: Optional[str] = None
    objective: Optional[str] = None
    agents: Optional[Union[List[AgentParams], AgentSwarm]] = None
    task_metadata: Optional[Dict[str, Any]] = None
    # metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    execution_metadata: Optional[Dict[str, Any]] = None
    # execution_mode: Literal["sequential", "parallel"] = "sequential"
    # stages: List[Stage] = Field(..., description="The sequence of stages that make up this task")


class WorkflowDefinition(BaseModel):
    """
    The top-level structure of the YAML.
    - name: A human-readable name for the workflow
    - agents: The agent definitions
    - tasks: The list of tasks that make up the workflow
    """

    name: str
    objective: Optional[str] = None  # Main text/prompt for the workflow
    client_mode: Optional[bool] = None
    agents: Optional[Union[List[AgentParams], AgentSwarm]] = None
    tasks: List[TaskDefinition] = Field(default_factory=list)
