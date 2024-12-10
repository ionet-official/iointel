from datetime import datetime
from datamodels import AgentParams
from models import Agent


def create_agent(params: AgentParams) -> Agent:
    """
    Create a Agent instance from the given AgentParams.
    """
    return Agent(name=params.name, instructions=params.instructions)


def get_current_datetime() -> str:
    """
    Return the current datetime as a string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")