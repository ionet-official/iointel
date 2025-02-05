import os

from .memory import Memory, AsyncMemory
from .agent_methods.data_models.datamodels import PersonaConfig

from langchain_openai import ChatOpenAI
import controlflow as cf
from typing import Optional, Callable

class Agent(cf.Agent):

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
        model: Optional[Callable] | Optional[str]= None,
        memories: Optional[list[Memory]] | Optional[list[AsyncMemory]]= None,
        interactive: Optional[bool] = False,
        llm_rules: Optional[cf.llm.rules.LLMRules] = None,
        **model_kwargs
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param description: A description of the agent. Visible to other agents.
        :param persona: A PersonaConfig instance to use for the agent. Used to set persona instructions.
        :param tools: A list of cf.Tool instances or @cf.tool decorated functions.
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
            model_instance = ChatOpenAI(model = model, **model_kwargs)

        elif model is not None:
            model_instance = model

        else:
            kwargs = dict(model_kwargs)
            for key, env_name in {
                "api_key": "OPENAI_API_KEY",
                "model": "OPENAI_API_MODEL",
                "base_url": "OPENAI_API_BASE_URL",
            }.items():
                if value := os.environ.get(env_name):
                    kwargs[key] = value
            model_instance = ChatOpenAI(**kwargs)

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
            memories=memories or [],
            interactive=interactive,
            llm_rules=llm_rules,

        )


    def run(self, prompt: str):
        return super().run(prompt)
    
    async def a_run(self, prompt: str):
        return await super().run_async(prompt)

    def set_instructions(self, new_instructions: str):
        self.instructions = new_instructions

    def add_tool(self, tool):
        updated_tools = self.tools + [tool]
        self.tools = updated_tools



