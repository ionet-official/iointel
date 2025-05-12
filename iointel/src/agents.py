from .memory import AsyncMemory
from .agent_methods.data_models.datamodels import PersonaConfig, Tool
from .utilities.rich import console
from .utilities.constants import get_api_url, get_base_model, get_api_key
from .utilities.registries import TOOLS_REGISTRY
from .utilities.helpers import supports_tool_choice_required

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent as PydanticAgent, Tool as PydanticTool

from pydantic import ConfigDict, SecretStr, BaseModel, ValidationError
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, ToolCallPart
from typing import Callable, Dict, Any, Optional, Union
import json
import dataclasses

import uuid
import asyncio

from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live


class PatchedValidatorTool(PydanticTool):
    _PATCH_ERR_TYPES = ("list_type",)

    async def run(self, message: ToolCallPart, *args, **kw):
        if (margs := message.args) and isinstance(margs, str):
            try:
                self._validator.validate_json(margs)
            except ValidationError as e:
                try:
                    margs_dict = json.loads(margs)
                except json.JSONDecodeError:
                    pass
                else:
                    patched = False
                    for err in e.errors():
                        if (
                            err["type"] in self._PATCH_ERR_TYPES
                            and len(err["loc"]) == 1
                            and err["loc"][0] in margs_dict
                            and isinstance(err["input"], str)
                        ):
                            try:
                                margs_dict[err["loc"][0]] = json.loads(err["input"])
                                patched = True
                            except json.JSONDecodeError:
                                pass
                    if patched:
                        message = dataclasses.replace(
                            message, args=json.dumps(margs_dict)
                        )
        return await super().run(message, *args, **kw)


class Agent(BaseModel):
    """
    A configurable agent that allows you to plug in different chat models,
    instructions, and tools. By default, it uses the pydantic OpenAIModel.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    name: str
    instructions: str
    persona: Optional[PersonaConfig] = None
    context: Optional[Any] = None
    tools: Optional[list] = None
    model: Optional[Union[OpenAIModel, str]] = None
    memory: Optional[AsyncMemory] = None
    model_settings: Optional[Dict[str, Any]] = (
        None  # dict(extra_body=None), #can add json model schema here
    )
    api_key: Optional[SecretStr | str] = None
    base_url: Optional[str] = None
    output_type: Optional[Any] = str
    _runner: PydanticAgent
    conversation_id: Optional[str] = None

    # args must stay in sync with AgentParams, because we use that model
    # to reconstruct agents
    def __init__(
        self,
        name: str,
        instructions: str,
        persona: Optional[PersonaConfig] = None,
        context: Optional[Any] = None,
        tools: Optional[list] = None,
        model: Optional[Union[OpenAIModel, str]] = None,
        memory: Optional[AsyncMemory] = None,
        model_settings: Optional[
            Dict[str, Any]
        ] = None,  # dict(extra_body=None), #can add json model schema here
        api_key: Optional[SecretStr | str] = None,
        base_url: Optional[str] = None,
        output_type: Optional[Any] = str,
        **model_kwargs,
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param description: A description of the agent. Visible to other agents.
        :param persona: A PersonaConfig instance to use for the agent. Used to set persona instructions.
        :param tools: A list of marvin.Tool instances or @marvin.fn decorated functions.
        :param model: A callable that returns a configured model instance.
                              If provided, it should handle all model-related configuration.
        :param model_kwargs: Additional keyword arguments passed to the model factory or ChatOpenAI if no factory is provided.

        If model_provider is given, you rely entirely on it for the model and ignore other model-related kwargs.
        If not, you fall back to ChatOpenAI with model_kwargs such as model="gpt-4o-mini", api_key="..."

        :param memory: A Memory instance to use for the agent. Memory module can store and retrieve data, and share context between agents.

        """
        resolved_api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(api_key or get_api_key())
        )
        resolved_base_url = base_url or get_api_url()

        if isinstance(model, OpenAIModel):
            resolved_model = model
        else:
            kwargs = dict(
                model_kwargs,
                provider=OpenAIProvider(
                    base_url=resolved_base_url, api_key=resolved_api_key.get_secret_value()
                ),
            )
            resolved_model = OpenAIModel(
                model_name=model if isinstance(model, str) else get_base_model(),
                **kwargs,
            )

        resolved_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, str):
                    registered_tool = TOOLS_REGISTRY.get(tool)
                elif isinstance(tool, Tool):
                    registered_tool = tool
                elif callable(tool):
                    registered_tool = Tool.from_function(tool)
                else:
                    raise ValueError(
                        f"Tool '{tool}' is neither a registered name nor a callable."
                    )
                if not registered_tool or not next(
                    (
                        name for name, t in TOOLS_REGISTRY.items()
                        if t.body == registered_tool.body
                    ),
                    None
                ):
                    raise ValueError(f"Tool '{tool}' not found in registry, did you forget to @register_tool?")
                resolved_tools.append(registered_tool.fn)

        if isinstance(model, str):
            model_supports_tool_choice = supports_tool_choice_required(model)
        else:
            model_supports_tool_choice = supports_tool_choice_required(
                getattr(model, "model_name", "")
            )

        if model_settings is None:
            model_settings = {}
        model_settings["supports_tool_choice_required"] = model_supports_tool_choice

        super().__init__(name=name, 
                         instructions=instructions, 
                         persona=persona, 
                         context=context, 
                         tools=resolved_tools, 
                         model=resolved_model, 
                         memory=memory, 
                         model_settings=model_settings, 
                         api_key=resolved_api_key, 
                         base_url=resolved_base_url, 
                         output_type=output_type)
        self._runner = PydanticAgent(
            name=name,
            tools=resolved_tools,
            model=resolved_model,
            model_settings=model_settings,
            output_type=output_type,
            end_strategy='exhaustive',
            retries=3,
            output_retries=5,
        )
        self._runner.system_prompt(dynamic=True)(self._make_init_prompt)

    def _make_init_prompt(self) -> str:
        # Combine user instructions with persona content
        combined_instructions = self.instructions
        # Build a persona snippet if provided
        if isinstance(self.persona, PersonaConfig):
            if persona_instructions := self.persona.to_system_instructions().strip():
                combined_instructions += "\n\n" + persona_instructions

        if self.context:
            combined_instructions += f"""\n\n 
            this is added context, perhaps a previous run, or anything else of value,
            so you can understand what is going on: {self.context}"""
        return combined_instructions

    def add_tool(self, tool):
        updated_tools = self.tools + [tool]
        self.tools = updated_tools

    async def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        pretty=True,
        message_history_limit=100,
        **kwargs,
    ):
        """
        Run the agent asynchronously.
        :param query: The query to run the agent on.
        :param conversation_id: The conversation ID to use for the agent.
        :param kwargs: Additional keyword arguments to pass to the agent.
        :return: The result of the agent run.
        """
        self.conversation_id = conversation_id

        if self.memory and conversation_id:
            try:
                stored_messages_json = await self.memory.get_message_history(
                    conversation_id, message_history_limit
                )
                if stored_messages_json:
                    message_history = stored_messages_json
                    kwargs["message_history"] = message_history
            except Exception as e:
                print("Error loading message history:", e)

        if not self.model_settings.get("supports_tool_choice_required"):
            if (output_type := kwargs.get("output_type")) is not None:
                if output_type is not str:
                    from pydantic_ai._output import (
                        get_union_args,
                        extract_str_from_union,
                    )

                    if union_args := get_union_args(output_type):
                        if not extract_str_from_union(output_type):
                            # if output_type is a Union but with no `str`, insert it there
                            output_type = Union[tuple(str, *union_args)]
                    else:
                        # it's not a Union and doesn't mention `str`, turn it into union with it
                        output_type = Union[str, output_type]
                    kwargs["output_type"] = output_type
        result = await self._runner.run(query, **kwargs)

        if self.memory:
            conversation_id = (
                conversation_id or self.conversation_id or str(uuid.uuid4())
            )

            try:
                await self.memory.store_run_history(conversation_id, result)
            except Exception as e:
                print("Error storing run history:", e)
        if pretty:
            task_header = Text(
                f" Objective: {query} ", style="bold white on dark_green"
            )
            agent_info = Text(f"Agent(s): {self.name}", style="cyan bold")
            result_info = Markdown(str(result.output), style="magenta")

            panel = Panel(
                result_info,
                title=task_header,
                subtitle=agent_info,
                border_style="electric_blue",
            )
            console.print(panel)

        return dict(
            result=result.output, conversation_id=conversation_id or self.conversation_id, full_result=result
        )

    async def run_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        return_markdown=False,
        message_history_limit=100,
        **kwargs,
    ):
        task_header = Text(f" Objective: {query} ", style="bold white on dark_green")
        agent_info = Text(f"Agent: {self.name}", style="cyan bold")
        markdown_content = ""

        if self.memory and conversation_id:
            try:
                stored_messages = await self.memory.get_message_history(
                    conversation_id, message_history_limit
                )
                if stored_messages:
                    kwargs["message_history"] = stored_messages
            except Exception as e:
                console.print(f"[red]Error loading message history:[/red] {e}")

        async with self._runner.iter(query, **kwargs) as agent_run:
            with Live(console=console, vertical_overflow="visible") as live:
                async for node in agent_run:
                    if self._runner.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    markdown_content += event.delta.content_delta or ""
                                    markdown_render = Markdown(
                                        markdown_content, style="magenta"
                                    )
                                    panel = Panel(
                                        markdown_render,
                                        title=task_header,
                                        subtitle=agent_info,
                                        border_style="electric_blue",
                                    )
                                    live.update(panel)

                # Explicitly ensure final markdown is exactly right:
                final_result = (
                    agent_run.result.output if hasattr(agent_run.result, "data") else ""
                )

                if not isinstance(final_result, str):
                    final_result_str = (
                        final_result.model_dump_json(indent=2)
                        if hasattr(final_result, "model_dump_json")
                        else str(final_result)
                    )
                else:
                    final_result_str = final_result

                # Safely compare lengths and update markdown_content
                if len(final_result_str) > len(markdown_content):
                    markdown_content += final_result_str[len(markdown_content) :]

                markdown_render = Markdown(markdown_content, style="magenta")
                panel = Panel(
                    markdown_render,
                    title=task_header,
                    subtitle=agent_info,
                    border_style="electric_blue",
                )
                live.update(panel)

                await asyncio.sleep(0.3)

        # Store updated conversation history.
        if self.memory:
            conversation_id = (
                conversation_id or self.conversation_id or str(uuid.uuid4())
            )
            try:
                await self.memory.store_run_history(
                    conversation_id, agent_run.result
                )
            except Exception as e:
                console.print(f"[red]Error storing run history:[/red] {e}")

        if return_markdown:
            result = dict(
                result=markdown_content,
                conversation_id=conversation_id or self.conversation_id,
            )
            return result
        return dict(
            result=agent_run.result.output,
            conversation_id=conversation_id or self.conversation_id,
            full_result=result
        )

    def set_context(self, context: Any):
        """
        Set the context for the agent.
        :param context: The context to set for the agent.
        """
        self.context = context

    @classmethod
    def make_default(cls):
        return cls(
            name="default-agent",
            instructions="you are a generalist who is good at everything.",
        )

