
from .memory import Memory, AsyncMemory
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

    def __init__(
        self,
        name: str,
        instructions: str,
        persona: Optional[PersonaConfig] = None,
        tools: Optional[list] = None,
        model: Optional[Union[OpenAIModel, str]] = None,
        memory: Optional[Union[Memory, AsyncMemory]] = None,
        model_settings: Optional[Dict[str, Any]] = dict(tool_choice="auto"), # FIXME: mutable default value is a bad thing
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

        :param memories: A list of Memory instances to use for the agent. Each memory module can store and retrieve data, and share context between agents.

        """

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

        # Build a persona snippet if provided
        if isinstance(persona, PersonaConfig):
            persona_instructions = persona.to_system_instructions()
        else:
            persona_instructions = ""

        # Combine user instructions with persona content
        combined_instructions = instructions
        if persona_instructions.strip():
            combined_instructions += "\n\n" + persona_instructions

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
        self.system_prompt = combined_instructions
        super().__init__(
            name=name,
            system_prompt=combined_instructions,
            tools=resolved_tools,
            model=model_instance,
            model_settings=model_settings,
            output_type=output_type,
        )

    def set_instructions(self, new_instructions: str):
        self.instructions = new_instructions

    def add_tool(self, tool):
        updated_tools = self.tools + [tool]
        self.tools = updated_tools


    def run_sync(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        pretty=True,
        message_history_limit=100,
        **kwargs,
    ):
        """
        Run the agent synchronously.
        :param query: The query to run the agent on.
        :param conversation_id: The conversation ID to use for the agent.
        :param kwargs: Additional keyword arguments to pass to the agent.
        :return: The result of the agent run.
        """
        self.conversation_id = conversation_id

        if self.memory and conversation_id:
            try:
                if isinstance(self.memory, AsyncMemory):
                    stored_messages = asyncio.run(
                        self.memory.get_message_history(
                            conversation_id, message_history_limit
                        )
                    )
                else:
                    stored_messages = self.memory.get_message_history(
                        conversation_id, message_history_limit
                    )
                if stored_messages:
                    kwargs["message_history"] = stored_messages
            except Exception as e:
                console.print(f"[red]Error loading message history:[/red] {e}")

        result = super().run_sync(query, **kwargs)

        if self.memory:
            conversation_id = (
                conversation_id or self.conversation_id or str(uuid.uuid4())
            )
            try:
                if isinstance(self.memory, AsyncMemory):
                    asyncio.run(self.memory.store_run_history(conversation_id, result))
                else:
                    self.memory.store_run_history(conversation_id, result)
            except Exception as e:
                console.print(f"[red]Error storing run history:[/red] {e}")

        if pretty:
            task_header = Text(
                f" Objective: {query} ", style="bold white on dark_green"
            )
            agent_info = Text(f"Agent(s): {self.name}", style="cyan bold")
            result_info = Markdown(f"{result.data}", style="magenta")

            # Display the result beautifully using Rich panel
            panel = Panel(
                result_info,
                title=task_header,
                subtitle=agent_info,
                border_style="electric_blue",
            )
            console.print(panel)

        return dict(
            result=result.data, conversation_id=conversation_id or self.conversation_id
        )

    async def run_async(
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
                if isinstance(self.memory, AsyncMemory):
                    stored_messages_json = await self.memory.get_message_history(
                        conversation_id, message_history_limit
                    )
                else:
                    stored_messages_json = (
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            self.memory.get_message_history,
                            conversation_id,
                            message_history_limit,
                        )
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
                if isinstance(self.memory, AsyncMemory):
                    await self.memory.store_run_history(conversation_id, result)
                else:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.memory.store_run_history, conversation_id, result
                    )
            except Exception as e:
                print("Error storing run history:", e)
        if pretty:
            task_header = Text(
                f" Objective: {query} ", style="bold white on dark_green"
            )
            agent_info = Text(f"Agent(s): {self.name}", style="cyan bold")
            result_info = Markdown(f"{result.data}", style="magenta")

            panel = Panel(
                result_info,
                title=task_header,
                subtitle=agent_info,
                border_style="electric_blue",
            )
            console.print(panel)

        return dict(
            result=result.data, conversation_id=conversation_id or self.conversation_id
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
                if isinstance(self.memory, AsyncMemory):
                    stored_messages = await self.memory.get_message_history(
                        conversation_id, message_history_limit
                    )
                else:
                    stored_messages = await asyncio.get_running_loop().run_in_executor(
                        None,
                        self.memory.get_message_history,
                        conversation_id,
                        message_history_limit,
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
                    agent_run.result.data if hasattr(agent_run.result, "data") else ""
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
                if isinstance(self.memory, AsyncMemory):
                    await self.memory.store_run_history(
                        conversation_id, agent_run.result
                    )
                else:
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        self.memory.store_run_history,
                        conversation_id,
                        agent_run.result,
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
            result=agent_run.result.data,
            conversation_id=conversation_id or self.conversation_id,
        )


    @classmethod
    def make_default(cls):
        return cls(
            name="default-agent",
            instructions="you are a generalist who is good at everything.",
            description="Default agent for tasks without agents",
        )


class Swarm(marvin.Swarm):
    def __init__(self, agents: List[Agent] = None, **kwargs):
        """
        :param agents: Optional list of Agent instances that this runner can orchestrate.
        """
        self.members = agents or []
        super().__init__(members=self.members, **kwargs)

    def __call__(self, agents: List[Agent] = None, **kwargs):
        self.members = agents or []
        return self
