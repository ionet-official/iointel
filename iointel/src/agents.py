import dataclasses
import json

from iointel.src.utilities.rich import pretty_output
from iointel.src.memory import AsyncMemory
from iointel.src.agent_methods.data_models.datamodels import (
    PersonaConfig,
    Tool,
    ToolUsageResult,
    AgentResult,
)
from iointel.src.utilities.rich import pretty_output
from iointel.src.utilities.helpers import supports_tool_choice_required, flatten_union_types
from iointel.src.ui.rich_panels import render_agent_result_panel
from iointel.src.ui.io_gradio_ui import IOGradioUI

from pydantic import ConfigDict, SecretStr, BaseModel, ValidationError
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, ToolCallPart
from typing import Callable, Dict, Any, Optional, Union, Literal, List

from pydantic_ai.models.openai import OpenAIModel, ModelRequestParameters
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent as PydanticAgent, Tool as PydanticTool
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings


class CleanedToolSchemaModel(OpenAIModel):
    """Custom model that cleans tool schemas to remove problematic fields like 'strict'."""
    
    def _clean_tool_schema(self, schema):
        """Clean tool schema by removing fields that are not accepted by the API."""
        if not isinstance(schema, dict):
            return schema
        
        # Create a deep copy to avoid modifying the original
        import copy
        cleaned_schema = copy.deepcopy(schema)
        
        # Remove 'strict' field from the schema
        if 'strict' in cleaned_schema:
            del cleaned_schema['strict']
        
        # Recursively clean nested objects
        for key, value in cleaned_schema.items():
            if isinstance(value, dict):
                cleaned_schema[key] = self._clean_tool_schema(value)
            elif isinstance(value, list):
                cleaned_schema[key] = [self._clean_tool_schema(item) if isinstance(item, dict) else item for item in value]
        
        return cleaned_schema
    
    async def request(self, messages, model_settings=None, model_request_parameters=None):
        """Override request to clean tool schemas and set tool_choice for GPT-OSS models."""
        # Debug: Print the model name being used
        print(f"🔍 CleanedToolSchemaModel.request() called with model_name: '{self.model_name}'")
        
        # Clean tool schemas in the request
        if model_request_parameters and hasattr(model_request_parameters, 'tools'):
            tools = model_request_parameters.tools
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and 'function' in tool:
                        if 'parameters' in tool['function']:
                            tool['function']['parameters'] = self._clean_tool_schema(tool['function']['parameters'])
        
        # Set tool_choice="auto" for GPT-OSS models (required by vLLM documentation)
        if isinstance(self.model_name, str) and "gpt-oss" in self.model_name.lower():
            print(f"🔧 Setting tool_choice='auto' for GPT-OSS model: {self.model_name}")
            if model_request_parameters is None:
                model_request_parameters = ModelRequestParameters()
            model_request_parameters.tool_choice = "auto"
        
        # Call the parent request method
        return await super().request(messages, model_settings, model_request_parameters)


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
        # HACK: Force load environment for Llama models BEFORE getting config
        if isinstance(model, str) and "llama" in model.lower():
            import os
            from dotenv import load_dotenv
            load_dotenv("creds.env", override=True)
            print(f"🔧 HACK: Force loading creds.env for Llama model: {model}")
            print(f"   IO_API_KEY present: {'IO_API_KEY' in os.environ}")
            print(f"   IO_API_BASE present: {'IO_API_BASE' in os.environ}")
            
            # Clear ALL caches so they re-read the env vars
            from .utilities.constants import get_api_url, get_api_key
            get_api_url.cache_clear()
            get_api_key.cache_clear()
        
        # Use centralized model configuration
        from .utilities.constants import get_model_config
        
        config = get_model_config(
            model=model if isinstance(model, str) else None,
            api_key=api_key if isinstance(api_key, str) else None,
            base_url=base_url
        )
        
        # HACK: Extra debug for Llama models
        if isinstance(model, str) and "llama" in model.lower():
            print(f"   Resolved API key: {config['api_key'][:20]}..." if config['api_key'] else "NONE")
            print(f"   Resolved base URL: {config['base_url']}")
        
        resolved_api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(config["api_key"])
        )
        resolved_base_url = config["base_url"]

        if isinstance(model, OpenAIModel):
            resolved_model = model
        else:
            model_name_to_use = model if isinstance(model, str) else "gpt-4o"
            
            # Check if this is a GPT-OSS model - enable tool calling with auto tool choice
            if isinstance(model, str) and "gpt-oss" in model.lower():
                print("🔧 Creating CleanedToolSchemaModel for GPT-OSS:")
                print(f"   model_name: {model_name_to_use}")
                print(f"   base_url: {resolved_base_url}")
                print(f"   api_key length: {len(resolved_api_key.get_secret_value())}")
                print("   ✅ GPT-OSS models: chat + tool calling enabled with auto tool choice")
                
                kwargs = dict(
                    model_kwargs,
                    provider=OpenAIProvider(
                        base_url=resolved_base_url,
                        api_key=resolved_api_key.get_secret_value(),
                    ),
                )
                resolved_model = CleanedToolSchemaModel(
                    model_name=model_name_to_use,
                    **kwargs,
                )
            else:
                # HACK: More debug for Llama models
                if isinstance(model, str) and "llama" in model.lower():
                    print("🔧 Creating CleanedToolSchemaModel:")
                    print(f"   model_name: {model}")
                    print(f"   base_url: {resolved_base_url}")
                    print(f"   api_key length: {len(resolved_api_key.get_secret_value())}")
                    
                kwargs = dict(
                    model_kwargs,
                    provider=OpenAIProvider(
                        base_url=resolved_base_url,
                        api_key=resolved_api_key.get_secret_value(),
                    ),
                )
                # Use CleanedToolSchemaModel to automatically clean tool schemas
                print(f"🔍 Creating CleanedToolSchemaModel with model_name: '{model_name_to_use}'")
                resolved_model = CleanedToolSchemaModel(
                    model_name=model_name_to_use,
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
        
        # For models that don't support tool choice, disable tools entirely
        if not model_supports_tool_choice:
            print(f"⚠️  Model '{model}' doesn't support tool choice - disabling tools")
            resolved_tools = []  # Disable tools for models that don't support tool choice

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
        print(f"--- Agent initialized with conversation_id: {self.conversation_id}")
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
        """Get a registered tool using the centralized tool registry utils."""
        from .utilities.tool_registry_utils import resolve_tool
        return resolve_tool(tool, allow_unregistered_tools)

    def _make_init_prompt(self) -> str:
        # Combine user instructions with persona content
        combined_instructions = self.instructions
        # Build a persona snippet if provided
        if isinstance(self.persona, PersonaConfig):
            if persona_instructions := self.persona.to_system_instructions().strip():
                combined_instructions += "\n\n" + persona_instructions

        if self.context:
            combined_instructions += f"""\n\n 
            this is added context, perhaps a previous run, tools, or anything else of value,
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
        # Debug logging to understand message structure
        if hasattr(self, 'debug') and self.debug:
            print(f"🔍 DEBUG: Extracting tool usage from {len(messages)} messages")
            for i, msg in enumerate(messages):
                print(f"  Message {i}: {type(msg)} - has parts: {hasattr(msg, 'parts')}")
                if hasattr(msg, "parts") and msg.parts:
                    for j, part in enumerate(msg.parts):
                        part_kind = getattr(part, "part_kind", None)
                        print(f"    Part {j}: {type(part)} - part_kind: {part_kind}")
        
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
        
        # Basic logging to understand what we found
        print(f"🔍 Found {len(tool_calls)} tool calls and {len(tool_returns)} tool returns")

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
    ) -> AgentResult:
        """
        Post-process agent result.
        """
        messages: list[ModelMessage] = (
            result.all_messages() if hasattr(result, "all_messages") else []
        )
        tool_usage_results = self.extract_tool_usage_results(messages)

        if pretty and (pretty_output is not None and pretty_output):
            render_agent_result_panel(
                result_output=result.output,
                query=query,
                agent_name=self.name,
                tool_usage_results=tool_usage_results,
                show_tool_calls=self.show_tool_calls,
                tool_pil_layout=self.tool_pil_layout,
            )

        return AgentResult(
            result=result.output,
            conversation_id=conversation_id,
            full_result=result,
            tool_usage_results=tool_usage_results,
        )


    def _resolve_conversation_id(self, conversation_id: Optional[str]) -> str | None:
        res = conversation_id or self.conversation_id or None
        print(f"--- Resolved conversation_id: {res}")
        return res

    async def _load_message_history(
        self, conversation_id: Optional[str], message_history_limit: int
    ) -> Optional[list[dict[str, Any]]]:
        convo_id = self._resolve_conversation_id(conversation_id)
        if self.memory and convo_id:
            try:
                messages = await self.memory.get_message_history(
                    convo_id, message_history_limit
                )
                print(f"Using memory: {bool(self.memory)}, loaded {len(messages) if messages else 0} messages")
                return messages
            except Exception as e:
                print("Error loading message history:", e)
        else:
            print(f"Using memory: {bool(self.memory)}, no messages loaded")
        return None

    def _adjust_output_type(self, kwargs: dict[str, Any]) -> None:
        """Adjusts the output type for models that don't support tool_choice_required.
        
        For models without tool_choice_required support, ensures str is included in the output
        type union to handle cases where the model returns a string response instead of using
        tools. This maintains backwards compatibility while allowing structured outputs.
        
        Args:
            kwargs: Dictionary of keyword arguments that may contain an output_type
        """
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
        pretty: Optional[bool] = None,
        message_history_limit=100,
        **kwargs,
    ) -> AgentResult:
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
                print(f"Storing run history for conversation {conversation_id}")
                stored = await self.memory.store_run_history(conversation_id, result)
                if stored:
                    print("Successfully stored run history")
                else:
                    print("Failed to store run history - store_run_history returned False")
            except Exception as e:
                print("Error storing run history:", e)

        return self._postprocess_agent_result(
            result, query, conversation_id, pretty=pretty
        )

    async def _stream_tokens(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        message_history_limit=100,
        **kwargs,
    ):
        """
        Async generator that yields partial content as tokens are streamed from the model.
        At the end, yields a dict with '__final__', 'content', and 'agent_result'.
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
            # After streaming, yield a special marker with the final result
            yield {
                "__final__": True,
                "content": content,
                "agent_result": agent_run.result,
            }

    async def run_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        return_markdown=False,
        message_history_limit=100,
        pretty: Optional[bool] = None,
        **kwargs,
    ) -> AgentResult:
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

        markdown_content = ""
        agent_result = None

        async for partial in self._stream_tokens(
            query,
            conversation_id=conversation_id,
            message_history_limit=message_history_limit,
            **kwargs,
        ):
            if isinstance(partial, dict) and partial.get("__final__"):
                markdown_content = partial["content"]
                agent_result = partial["agent_result"]
                break
            else:
                markdown_content = partial  # or accumulate if you want

        conversation_id = self._resolve_conversation_id(conversation_id)
        if self.memory:
            try:
                await self.memory.store_run_history(conversation_id, agent_result)
            except Exception as e:
                print("Error storing run history:", e)

        result = self._postprocess_agent_result(
            agent_result, query, conversation_id, pretty=pretty
        )
        if return_markdown:
            result.result = markdown_content
        return result

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
        self, interface_title: Optional[str] = None, share: bool = False, conversation_id: Optional[str] = None
    ) -> None:
        """
        Launches a Gradio UI for interacting with the agent as a chat interface.
        """
        ui = IOGradioUI(agent=self, interface_title=interface_title, conversation_id=conversation_id)
        return await ui.launch(share=share)


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
        show_tool_calls: bool = True,
        tool_pil_layout: Literal["vertical", "horizontal"] = "horizontal",
        debug: bool = False,
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
            show_tool_calls=show_tool_calls,
            tool_pil_layout=tool_pil_layout,
            debug=debug,
            **model_kwargs,
        )
