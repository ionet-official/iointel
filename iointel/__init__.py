from .src.utilities.magic import (
    UNUSED as UNUSED,
)  # this performs some magic to hide controlflow warning
from .src.agents import Agent

from .src.workflow import Workflow, run_agents
from .src.agent_methods.data_models.datamodels import PersonaConfig
from .client import client


__all__ = [
    "Agent",
    "Workflow",
    "run_agents",
    "PersonaConfig",
    "client",
]


__version__ = "1.1.3"
