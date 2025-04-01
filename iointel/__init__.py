from .src.utilities.magic import (
    UNUSED as UNUSED,
)  # this performs some magic to stop prefect mucking with logging setup
from .src.agents import Agent, Swarm

# from .src.memory import AsyncMemory, AsyncPostgresMemoryProvider, Memory, PostgresMemoryProvider
from .src.memory import Memory, PostgresMemoryProvider

# from .src.task import Tasks
from .src.workflow import Workflow
from .src.agent_methods.data_models.datamodels import PersonaConfig

# from .src.handlers import AsyncLoggingHandler, LoggingHandler
from .src.utilities.decorators import register_custom_task, register_tool
from .src.utilities.runners import run_agents, run_agents_async
from .src.code_parsers.pycode_parser import PythonModule, ClassDefinition, FunctionDefinition, Argument, ImportStatement , PythonCodeGenerator

__all__ = [


    ###agents###
    "Agent",
    "Swarm",

    ###memory###
    "Memory",
    "PostgresMemoryProvider",
    # "AsyncMemory",
    # "AsyncPostgresMemoryProvider",

    ##workflows and runners and registries##
    "Workflow",
    "register_custom_task",
    "register_tool",
    "run_agents",
    "run_agents_async",

    ###persona####
    "PersonaConfig",

    # "AsyncLoggingHandler",
    # "LoggingHandler"

    ###code parsers###
    "PythonModule",
    "ClassDefinition",
    "FunctionDefinition",
    "Argument",
    "ImportStatement",
    "PythonCodeGenerator"


]


__version__ = "1.1.3"
