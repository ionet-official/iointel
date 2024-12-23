from datetime import datetime
from framework.src.agent_methods.data_models.datamodels import AgentParams
from framework.src.agents import Agent


def create_agent(params: AgentParams) -> Agent:
    """
    Create a Agent instance from the given AgentParams.
    """
    return Agent(name=params.name, instructions=params.instructions, tools=params.tools)


def get_current_datetime() -> str:
    """
    Return the current datetime as a string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")