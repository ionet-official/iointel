# from .memory import Memory, AsyncMemory

from .memory import Memory
from .agent_methods.data_models.datamodels import PersonaConfig
from iointel.src.constants import get_api_url, get_base_model, get_api_key

from pydantic_ai.models.openai import OpenAIModel
from pydantic import SecretStr
import marvin
from typing import List, Dict, Any, Optional, Union
from prefect import task

import os


class Agent(marvin.Agent):
    """
    A configurable agent that allows you to plug in different chat models,
    instructions, and tools. By default, it uses the pydantic OpenAIModel.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        description: Optional[str] = None,
        persona: Optional[PersonaConfig] = None,
        tools: Optional[list] = None,
        model: Optional[Union[OpenAIModel, str]] = None,
        memories: Optional[list[Memory]] = None,
        model_settings: Optional[Dict[str, Any]] = dict(tool_choice="auto"),
        api_key: Optional[SecretStr] = None,
        base_url: Optional[str] = None,
        **model_kwargs,
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param description: A description of the agent. Visible to other agents.
        :param persona: A PersonaConfig instance to use for the agent. Used to set persona instructions.
        :param tools: A list of cf.Tool instances or @cf.tool decorated functions.
        :param model: A callable that returns a configured model instance.
                              If provided, it should handle all model-related configuration.
        :param model_kwargs: Additional keyword arguments passed to the model factory or ChatOpenAI if no factory is provided.

        If model_provider is given, you rely entirely on it for the model and ignore other model-related kwargs.
        If not, you fall back to ChatOpenAI with model_kwargs such as model="gpt-4o-mini", api_key="..."

        :param memories: A list of Memory instances to use for the agent. Each memory module can store and retrieve data, and share context between agents.

        """
        self.api_key = SecretStr(api_key)
        self.base_url = base_url

        if isinstance(model, str):
            model_instance = OpenAIModel(
                model_name=model,
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
            )

        elif isinstance(model, OpenAIModel):
            model_instance = model

        else:
            kwargs = dict(model_kwargs)
            for key, value in [
                ("api_key", get_api_key()),
                ("model", get_base_model()),
                ("base_url", get_api_url()),
            ]:
                if value:
                    kwargs[key] = value
            if self.api_key:
                kwargs["api_key"] = self.api_key.get_secret_value()
            if self.base_url:
                kwargs["base_url"] = base_url
            model_instance = OpenAIModel(**kwargs)


        # Build a persona snippet if provided
        if isinstance(persona, PersonaConfig):
            persona_instructions = persona.to_system_instructions()
        else:
            persona_instructions = ""

        # Combine user instructions with persona content
        combined_instructions = instructions
        if persona_instructions.strip():
            combined_instructions += "\n\n" + persona_instructions


        super().__init__(
            name=name,
            instructions=combined_instructions,
            description=description,
            tools=tools or [],
            model=model_instance,
            memories=memories or [],
            model_settings=model_settings,
        )

    @task(persist_result=False)
    def run(self, prompt: str):
        return super().run(prompt)

    @task(persist_result=False)
    async def a_run(self, prompt: str):
        return await super().run_async(prompt)

    def set_instructions(self, new_instructions: str):
        self.instructions = new_instructions

    def add_tool(self, tool):
        updated_tools = self.tools + [tool]
        self.tools = updated_tools


class Swarm(marvin.Swarm):
    def __init__(self, agents: List[Agent] = None, **kwargs):
        self.members = agents or []
        """
            :param agents: Optional list of Agent instances that this runner can orchestrate.
            """
        super().__init__(members=self.members, **kwargs)

    def __call__(self, agents: List[Agent] = None, **kwargs):
        self.members = agents or []
        return self
