from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import controlflow as cf
import os

load_dotenv()


class Agent:

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
        tools: list = None,
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        model_type: str = "openai",  # could be "llama" or other in the future
        **model_kwargs
    ):
        """
        :param name: The name of the agent.
        :param instructions: The instruction prompt for the agent.
        :param tools: A list of cf.Tool instances or @cf.tool decorated functions.
        :param model_name: The name of the model (e.g. "gpt-4", "gpt-3.5-turbo").
        :param api_key: The API key for the chosen model. Defaults to OPENAI_API_KEY from environment.
        :param model_type: The type of model backend ("openai", "llama", etc.).
        :param model_kwargs: Additional keyword arguments passed to the model constructor.
        """

        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Set up the model interface depending on model_type
        if self.model_type == "openai":
            self.model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                **model_kwargs
            )
        elif self.model_type == "llama":
            # Placeholder for Llama model integration:
            # self.model = LlamaChat(model=self.model_name, api_key=self.api_key, **model_kwargs)
            raise NotImplementedError("Llama model integration not yet implemented.")
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Create the cf.Agent using the configured model, instructions, and tools
        self.agent = cf.Agent(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model
        )

    def run(self, prompt: str):
        """
        Runs the agent on a given prompt. This uses cf.Agent's run method to produce a response.
        """
        return self.agent.run(prompt)

    def set_instructions(self, new_instructions: str):
        """
        Update the agent's instructions at runtime if needed.
        """
        self.instructions = new_instructions
        self.agent.instructions = new_instructions

    def add_tool(self, tool):
        """
        Add a new tool to the agent's toolkit dynamically.
        """
        self.tools.append(tool)
        self.agent.tools = self.tools

    def set_model(self, model_name: str, model_type: str = "openai", **model_kwargs):
        """
        Switch the underlying model at runtime, if desired. This will recreate the agent
        with the new model configuration.
        """
        self.model_name = model_name
        self.model_type = model_type
        if self.model_type == "openai":
            self.model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                **model_kwargs
            )
        elif self.model_type == "llama":
            # Placeholder for Llama model integration
            raise NotImplementedError("Llama model integration not yet implemented.")
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.agent = cf.Agent(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            model=self.model
        )