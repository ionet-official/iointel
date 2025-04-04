from .src.utilities.magic import (
    UNUSED as UNUSED,
)  # this performs some magic to stop prefect mucking with logging setup
from .src.agents import Agent, Swarm
from .src.memory import Memory
from .src.workflow import Workflow
from .src.utilities.runners import run_agents, run_agents_async
from .src.agent_methods.data_models.datamodels import PersonaConfig
from .src.utilities.decorators import register_custom_task, register_tool

from .src.code_parsers.pycode_parser import (
    PythonModule,
    ClassDefinition,
    FunctionDefinition,
    Argument,
    ImportStatement,
    PythonCodeGenerator,
)

__all__ = [
    ###agents###
    "Agent",
    "Swarm",
    "Memory",
    "Workflow",
    "register_custom_task",
    "register_tool",
    "run_agents",
    "run_agents_async",
    "register_custom_task",
    "register_tool",
    "PersonaConfig",
    ###code parsers###
    "PythonModule",
    "ClassDefinition",
    "FunctionDefinition",
    "Argument",
    "ImportStatement",
    "PythonCodeGenerator",
]


__version__ = "1.1.3"
