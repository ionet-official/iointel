
from .agent_methods.data_models.datamodels import PersonaConfig
from .utilities.constants import get_api_url, get_base_model, get_api_key

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import marvin

from typing import Optional, Callable


class Agent(marvin.Agent):
    """
    A configurable wrapper around cf.Agent that allows you to plug in different chat models,
    instructions, and tools. By default, it uses the ChatOpenAI model.

    In the future, you can add logic to switch to a Llama-based model or other models by
    adding conditionals or separate model classes.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        description: Optional[str] = None,
        persona: Optional[PersonaConfig] = None,
        tools: Optional[list] = None,
        model: Optional[Callable] | Optional[str] = None,
        **model_kwargs,
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param description: A description of the agent. Visible to other agents.
        :param persona: A PersonaConfig instance to use for the agent. Used to set persona instructions.
        :param tools: A list of marvin.Tool instances or @marvin.fn decorated functions.
        :param model_provider: A callable that returns a configured model instance.
                              If provided, it should handle all model-related configuration.
        :param model_kwargs: Additional keyword arguments passed to the model factory or ChatOpenAI if no factory is provided.

        If model_provider is given, you rely entirely on it for the model and ignore other model-related kwargs.
        If not, you fall back to ChatOpenAI with model_kwargs such as model="gpt-4o-mini", api_key="..."

        :param memories: A list of Memory instances to use for the agent. Each memory module can store and retrieve data, and share context between agents.
        :param interactive: A boolean flag to indicate if the agent is interactive. If True, the agent can run in interactive mode.
        :param llm_rules: An LLMRules instance to use for the agent. If provided, the agent uses the LLMRules for logic-based reasoning.

        """
        if isinstance(model, str):
            model_instance = OpenAIModel(model_name=model, **model_kwargs)

        elif model is not None:
            model_instance = model

        else:
            kwargs = dict(model_kwargs)
            for key, value in [
                ("model_name", get_base_model()),
                ("provider", OpenAIProvider(
                    base_url=get_api_url(),
                    api_key=get_api_key()
                )),
            ]:
                if value:
                    kwargs[key] = value
            model_instance = OpenAIModel(**kwargs)

        # Build a persona snippet if provided
        persona_instructions = ""
        if persona:
            persona_instructions = persona.to_system_instructions()

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
        )

    def get_end_turn_tools(self):
        return [str] + super().get_end_turn_tools()  # a hack to override tool_choice='auto'

    def run(self, prompt: str):
        return super().run(prompt)

    async def a_run(self, prompt: str):
        return await super().run_async(prompt)

    def set_instructions(self, new_instructions: str):
        self.instructions = new_instructions

    def add_tool(self, tool):
        updated_tools = self.tools + [tool]
        self.tools = updated_tools
