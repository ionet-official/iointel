from datetime import datetime
from datamodels import AgentParams
import time
import controlflow as cf


def remind_after_delay(command: str, delay: int = 0) -> str:
    """
    A simple task that waits for `delay` seconds and then returns a reminder message.
    """
    if delay > 0:
        time.sleep(delay)
    return f"Reminder: {command}"

def create_agent(params: AgentParams) -> cf.Agent:
    """
    Create a controlflow.Agent instance from the given AgentParams.
    """
    return cf.Agent(name=params.name, instructions=params.instructions)


def get_current_datetime() -> str:
    """
    Return the current datetime as a string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")