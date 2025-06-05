from .memory import AsyncMemory
from .agent_methods.data_models.datamodels import PersonaConfig, Tool, ToolUsageResult
from .utilities.rich import pretty_output
from .utilities.constants import get_api_url, get_base_model, get_api_key
from .utilities.registries import TOOLS_REGISTRY
from .utilities.helpers import supports_tool_choice_required, flatten_union_types
from .ui.io_gradio_ui import IOGradioUI

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent as PydanticAgent, Tool as PydanticTool
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings

from pydantic import ConfigDict, SecretStr, BaseModel, ValidationError
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, ToolCallPart
from typing import Callable, Dict, Any, Optional, Union, Literal
import json
import dataclasses

import uuid
import asyncio


class PatchedValidatorTool(PydanticTool):
    _PATCH_ERR_TYPES = ("list_type",)

    async def run(self, message: ToolCallPart, *args, **kw):
        if (margs := message.args) and isinstance(margs, str):
            try:
                self.function_schema.validator.validate_json(margs)
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
    model_settings: Optional[ModelSettings | Dict[str, Any]] = (
        None  # dict(extra_body=None), #can add json model schema here
    )
    api_key: SecretStr
    base_url: Optional[str] = None
    output_type: Optional[Any] = str
    _runner: PydanticAgent
    conversation_id: Optional[str] = None
    show_tool_calls: bool = True
    tool_pil_layout: Literal["vertical", "horizontal"] = (
        "horizontal"  # 'vertical' or 'horizontal'
    )
    debug: bool = False
    _allow_unregistered_tools: bool

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
            ModelSettings | Dict[str, Any]
        ] = None,  # dict(extra_body=None), #can add json model schema here
        api_key: Optional[SecretStr | str] = None,
        base_url: Optional[str] = None,
        output_type: Optional[Any] = str,
        conversation_id: Optional[str] = None,
        retries: int = 3,
        output_retries: int | None = None,
        show_tool_calls: bool = True,
        tool_pil_layout: Literal["vertical", "horizontal"] = "horizontal",
        debug: bool = False,
        allow_unregistered_tools: bool = False,
        **model_kwargs,
    ) -> None:
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param description: A description of the agent. Visible to other agents.
        :param persona: A PersonaConfig instance to use for the agent. Used to set persona instructions.
        :param tools: A list of Tool instances or @register_tool decorated functions.
        :param model: A callable that returns a configured model instance.
                              If provided, it should handle all model-related configuration.
        :param model_kwargs: Additional keyword arguments passed to the model factory or ChatOpenAI if no factory is provided.
        :param verbose: If True, displays detailed tool usage information during execution.
        :param tool_pil_layout: 'horizontal' (default) or 'vertical' for tool PIL stacking.

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
                    base_url=resolved_base_url,
                    api_key=resolved_api_key.get_secret_value(),
                ),
            )
            resolved_model = OpenAIModel(
                model_name=model if isinstance(model, str) else get_base_model(),
                **kwargs,
            )

        resolved_tools = [
            self._get_registered_tool(tool, allow_unregistered_tools)
            for tool in (tools or ())
        ]

        if isinstance(model, str):
            model_supports_tool_choice = supports_tool_choice_required(model)
        else:
            model_supports_tool_choice = supports_tool_choice_required(
                getattr(model, "model_name", "")
            )

        if model_settings is None:
            model_settings = {}
        model_settings["supports_tool_choice_required"] = model_supports_tool_choice

        super().__init__(
            name=name,
            instructions=instructions,
            persona=persona,
            context=context,
            tools=resolved_tools,
            model=resolved_model,
            memory=memory,
            model_settings=model_settings,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            output_type=output_type,
            show_tool_calls=show_tool_calls,
            tool_pil_layout=tool_pil_layout,
            debug=debug,
            conversation_id=conversation_id,
        )
        self._allow_unregistered_tools = allow_unregistered_tools
        self._runner = PydanticAgent(
            name=name,
            tools=[
                PatchedValidatorTool(
                    fn.get_wrapped_fn(), name=fn.name, description=fn.description
                )
                for fn in resolved_tools
            ],
            model=resolved_model,
            model_settings=model_settings,
            output_type=output_type,
            end_strategy="exhaustive",
            retries=retries,
            output_retries=output_retries,
        )
        self._runner.system_prompt(dynamic=True)(self._make_init_prompt)

    @classmethod
    def _get_registered_tool(
        cls, tool: str | Tool | Callable, allow_unregistered_tools: bool
    ) -> Tool:
        if isinstance(tool, str):
            if not (registered_tool := TOOLS_REGISTRY.get(tool)):
                raise ValueError(
                    f"Tool '{tool}' not found in registry, did you forget to @register_tool?"
                )
        elif isinstance(tool, Tool):
            registered_tool = tool
        elif callable(tool):
            registered_tool = Tool.from_function(tool)
        else:
            raise ValueError(
                f"Tool '{tool}' is neither a registered name nor a callable."
            )
        found_tool = next(
            (
                tool
                for tool in TOOLS_REGISTRY.values()
                if tool.body == registered_tool.body
            ),
            None,
        )
        if not found_tool:
            if allow_unregistered_tools:
                found_tool = registered_tool
            else:
                raise ValueError(
                    f"Tool '{registered_tool.name}' not found in registry, did you forget to @register_tool?"
                )
        # we need to take tool name and description from the registry,
        # as the user might have passed in an underlying function
        # instead of the registered tool object
        return registered_tool.model_copy(
            update={"name": found_tool.name, "description": found_tool.description}
        )

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
        registered_tool = self._get_registered_tool(
            tool, self._allow_unregistered_tools
        )
        self.tools += [registered_tool]
        self._runner._register_tool(
            PatchedValidatorTool(registered_tool.get_wrapped_fn())
        )

    def extract_tool_usage_results(
        self, messages: list[ModelMessage]
    ) -> list[ToolUsageResult]:
        """
        Given a list of messages, extract ToolUsageResult objects.
        Handles multiple tool calls/returns per message.
        Returns a list of ToolUsageResult.
        """
        # Collect all tool-calls and tool-returns by tool_call_id
        tool_calls = {}
        tool_returns = {}

        for msg in messages:
            if not hasattr(msg, "parts") or not msg.parts:
                continue
            for part in msg.parts:
                if getattr(part, "part_kind", None) == "tool-call":
                    tool_call_id = getattr(part, "tool_call_id", None)
                    if tool_call_id:
                        tool_calls[tool_call_id] = part
                elif getattr(part, "part_kind", None) == "tool-return":
                    tool_call_id = getattr(part, "tool_call_id", None)
                    if tool_call_id:
                        tool_returns[tool_call_id] = part

        tool_usage_results: list[ToolUsageResult] = []

        # Pair tool-calls with their returns
        for tool_call_id, call_part in tool_calls.items():
            tool_name = getattr(call_part, "tool_name", None)
            tool_args = getattr(call_part, "args", {})
            # Try to parse args if it's a JSON string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            tool_result = None
            if tool_call_id in tool_returns:
                tool_result = getattr(tool_returns[tool_call_id], "content", None)
            tool_usage_results.append(
                ToolUsageResult(
                    tool_name=tool_name,
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                    tool_result=tool_result,
                )
            )
        return tool_usage_results

    def _postprocess_agent_result(
        self,
        result: AgentRunResult,
        query: str,
        conversation_id: Union[str, int],
        pretty: bool = True,
    ):
        messages: list[ModelMessage] = (
            result.all_messages() if hasattr(result, "all_messages") else []
        )
        tool_usage_results = self.extract_tool_usage_results(messages)

        if pretty and (pretty_output is not None and pretty_output):
            from iointel.src.ui.rich_panels import render_agent_result_panel

            render_agent_result_panel(
                result_output=result.output,
                query=query,
                agent_name=self.name,
                tool_usage_results=tool_usage_results,
                show_tool_calls=self.show_tool_calls,
                tool_pil_layout=self.tool_pil_layout,
            )
        return dict(
            result=result.output,
            conversation_id=conversation_id,
            full_result=result,
            tool_usage_results=tool_usage_results,
        )

    async def _load_message_history(
        self, conversation_id: Optional[str], message_history_limit: int
    ) -> Optional[list[dict[str, Any]]]:
        if self.memory and conversation_id:
            try:
                return await self.memory.get_message_history(
                    conversation_id, message_history_limit
                )
            except Exception as e:
                print("Error loading message history:", e)
        return None

    def _resolve_conversation_id(self, conversation_id: Optional[str]) -> str:
        return conversation_id or self.conversation_id or str(uuid.uuid4())

    def _adjust_output_type(self, kwargs: dict[str, Any]) -> None:
        if not self.model_settings.get("supports_tool_choice_required"):
            output_type = kwargs.get("output_type")
            if output_type is not None and output_type is not str:
                flat_types = flatten_union_types(output_type)
                if str not in flat_types:
                    flat_types = [str] + flat_types
                kwargs["output_type"] = Union[tuple(flat_types)]

    async def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        pretty: bool = True,
        message_history_limit=100,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run the agent asynchronously.
        :param query: The query to run the agent on.
        :param conversation_id: The conversation ID to use for the agent.
        :param pretty: Whether to pretty print the result as a rich panel, useful for cli or notebook.
        :param message_history_limit: The number of messages to load from the memory.
        :param kwargs: Additional keyword arguments to pass to the agent.
        :return: The result of the agent run.
        """

        message_history = await self._load_message_history(
            conversation_id, message_history_limit
        )
        if message_history:
            kwargs["message_history"] = message_history

        self._adjust_output_type(kwargs)

        result = await self._runner.run(query, **kwargs)

        conversation_id = self._resolve_conversation_id(conversation_id)

        if self.memory:
            try:
                await self.memory.store_run_history(conversation_id, result)
            except Exception as e:
                print("Error storing run history:", e)

        return self._postprocess_agent_result(
            result, query, conversation_id, pretty=pretty
        )

    async def run_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        return_markdown=False,
        message_history_limit=100,
        pretty: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run the agent with streaming output.
        :param query: The query to run the agent on.
        :param conversation_id: The optional conversation ID to use for the agent.
        :param return_markdown: Whether to return the result as markdown.
        :param message_history_limit: The number of messages to load from the memory.
        :param pretty: Whether to pretty print the result as a rich panel, useful for cli or notebook.
        :param kwargs: Additional keyword arguments to pass to the agent.
        :return: The result of the agent run.
        """
        from iointel.src.ui.rich_panels import update_live_panel
        from iointel.src.utilities.rich import console
        from rich.live import Live

        markdown_content = ""
        message_history = await self._load_message_history(
            conversation_id, message_history_limit
        )
        if message_history:
            kwargs["message_history"] = message_history

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
                                    update_live_panel(
                                        live, markdown_content, query, self.name
                                    )

                final_result = (
                    agent_run.result.output if hasattr(agent_run.result, "data") else ""
                )
                final_result_str = (
                    (
                        final_result.model_dump_json(indent=2)
                        if hasattr(final_result, "model_dump_json")
                        else str(final_result)
                    )
                    if not isinstance(final_result, str)
                    else final_result
                )

                if len(final_result_str) > len(markdown_content):
                    markdown_content += final_result_str[len(markdown_content) :]

                update_live_panel(live, markdown_content, query, self.name)
                await asyncio.sleep(0.3)

        conversation_id = self._resolve_conversation_id(conversation_id)
        if self.memory:
            try:
                await self.memory.store_run_history(conversation_id, agent_run.result)
            except Exception as e:
                console.print(f"[red]Error storing run history:[/red] {e}")

        result_dict = self._postprocess_agent_result(
            agent_run.result, query, conversation_id, pretty=pretty
        )
        if return_markdown:
            result_dict["result"] = markdown_content
        return result_dict

    def set_context(self, context: Any) -> None:
        """
        Set the context for the agent.
        :param context: The context to set for the agent.
        """
        self.context = context

    @classmethod
    def make_default(cls) -> "Agent":
        return cls(
            name="default-agent",
            instructions="you are a generalist who is good at everything.",
        )

    async def get_conversation_ids(self) -> list[str]:
        if hasattr(self, "memory") and self.memory:
            try:
                convos = await self.memory.list_conversation_ids()
                return convos or []
            except Exception as e:
                print(f"Error fetching conversation IDs: {e}")
        return []

    async def launch_chat_ui(
        self, interface_title: str = None, share: bool = False
    ) -> None:
        """
        Launches a Gradio UI for interacting with the agent as a chat interface.
        """
        ui = IOGradioUI(agent=self, interface_title=interface_title)
        return await ui.launch(share=share)

    async def stream_tokens(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        message_history_limit=100,
        **kwargs,
    ):
        """
        Async generator that yields partial content as tokens are streamed from the model.
        """
        message_history = await self._load_message_history(
            conversation_id, message_history_limit
        )
        if message_history:
            kwargs["message_history"] = message_history

        async with self._runner.iter(query, **kwargs) as agent_run:
            content = ""
            async for node in agent_run:
                if self._runner.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                content += event.delta.content_delta or ""
                                yield content


class LiberalToolAgent(Agent):
    """
    A subclass of iointel.Agent that allows passing in arbitrary callables as tools
    without requiring one to register them first
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        persona: Optional[PersonaConfig] = None,
        context: Optional[Any] = None,
        tools: Optional[list] = None,
        model: Optional[Union[OpenAIModel, str]] = None,
        memory: Optional[AsyncMemory] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        api_key: Optional[SecretStr | str] = None,
        base_url: Optional[str] = None,
        output_type: Optional[Any] = str,
        retries: int = 3,
        output_retries: int | None = None,
        **model_kwargs,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            persona=persona,
            context=context,
            tools=tools,
            model=model,
            memory=memory,
            model_settings=model_settings,
            api_key=api_key,
            base_url=base_url,
            output_type=output_type,
            retries=retries,
            output_retries=output_retries,
            allow_unregistered_tools=True,
            **model_kwargs,
        )
