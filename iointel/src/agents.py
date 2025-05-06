
from .memory import AsyncMemory
from .agent_methods.data_models.datamodels import PersonaConfig
from .utilities.rich import console
from .utilities.constants import get_api_url, get_base_model, get_api_key
from .utilities.registries import TOOLS_REGISTRY

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent as PydanticAgent

from pydantic import SecretStr
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
from typing import Dict, Any, Optional, Union

import uuid
import asyncio

from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live


class Agent(PydanticAgent):
    """
    A configurable agent that allows you to plug in different chat models,
    instructions, and tools. By default, it uses the pydantic OpenAIModel.
    """

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
        model_settings: Optional[Dict[str, Any]] = None, #dict(extra_body=None), #can add json model schema here
        api_key: Optional[SecretStr | str] = None,
        base_url: Optional[str] = None,
        result_type: Optional[Any] = str,
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
        # save some parameters for later dumping to AgentParams
        self._instructions = instructions
        self._context = context
        self._persona = persona

        self.api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(api_key or get_api_key())
        )
        self.base_url = base_url or get_api_url()

        if isinstance(model, OpenAIModel):
            model_instance = model

        else:
            kwargs = dict(
                model_kwargs,
                provider=OpenAIProvider(
                    base_url=self.base_url, api_key=self.api_key.get_secret_value()
                ),
            )
            model_instance = OpenAIModel(
                model_name=model if isinstance(model, str) else get_base_model(),
                **kwargs,
            )

        self.memory = memory

        resolved_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, str):
                    registered_tool = TOOLS_REGISTRY.get(tool)
                    if not registered_tool:
                        raise ValueError(f"Tool '{tool}' not found in registry.")
                    resolved_tools.append(registered_tool.fn)
                elif callable(tool):
                    resolved_tools.append(tool)
                else:
                    raise ValueError(
                        f"Tool '{tool}' is neither a registered name nor a callable."
                    )

        self.tools = resolved_tools
        super().__init__(
            name=name,
            tools=resolved_tools,
            model=model_instance,
            model_settings=model_settings,
            result_type=result_type,
        )

        self.system_prompt(dynamic=True)(self._make_init_prompt)


    def _make_init_prompt(self) -> str:
        # Build a persona snippet if provided
        if isinstance(self._persona, PersonaConfig):
            persona_instructions = self._persona.to_system_instructions().strip()
        else:
            persona_instructions = ""
        # Combine user instructions with persona content
        combined_instructions = self._instructions
        if persona_instructions:
            combined_instructions += "\n\n" + persona_instructions

        if self._context:
            combined_instructions += f"""\n\n 
            this is added context, 
            perhaps a previous run, 
            or anything else of value,
            so you can understand whats going on: {self._context}"""
        return combined_instructions

    @property
    def instructions(self):
        return self._instructions

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

        result = await super().run(query, **kwargs)

        if self.memory:
            conversation_id = (
                conversation_id or self.conversation_id or str(uuid.uuid4())
            )

            try:
                await self.memory.store_run_history(conversation_id, result)
            except Exception as e:
                print("Error storing run history:", e)
        # pydantic is planning to rename a field, try both for compatibility
        result_output = getattr(result, "output", None) or getattr(result, "data", None)
        if pretty:
            task_header = Text(
                f" Objective: {query} ", style="bold white on dark_green"
            )
            agent_info = Text(f"Agent(s): {self.name}", style="cyan bold")
            result_info = Markdown(result_output, style="magenta")

            panel = Panel(
                result_info,
                title=task_header,
                subtitle=agent_info,
                border_style="electric_blue",
            )
            console.print(panel)

        return dict(
            result=result_output, conversation_id=conversation_id or self.conversation_id, full_result=result
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

        async with self.iter(query, **kwargs) as agent_run:
            with Live(console=console, vertical_overflow="visible") as live:
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
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

        self._context = context

    @classmethod
    def make_default(cls):
        return cls(
            name="default-agent",
            instructions="you are a generalist who is good at everything.",
        )

