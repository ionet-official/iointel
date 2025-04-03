from typing import Callable
from prefect import task
from .registries import (
    CHAINABLE_METHODS,
    TASK_EXECUTOR_REGISTRY,
    CUSTOM_WORKFLOW_REGISTRY,
    TOOLS_REGISTRY,
)
from ..workflow import Workflow
from ..agent_methods.data_models.datamodels import Tool
from ..utilities.helpers import make_logger


logger = make_logger(__name__)


########register custom task decorator########
def register_custom_task(task_type: str, chainable: bool = True):
    """
    Decorator that registers a custom task executor for a given task type.
    Additionally, if chainable=True (default), it creates and attaches a chainable
    method to the Tasks class so that it can be called as tasks.<task_type>(**kwargs).
    """

    def decorator(executor_fn: Callable):
        # Register the executor function for later task execution.
        prefect_task = task(executor_fn, name=task_type, persist_result=False)
        TASK_EXECUTOR_REGISTRY[task_type] = prefect_task

        if chainable:

            def chainable_method(self, **kwargs):
                # Create a task dictionary for this custom task.
                task_dict = {
                    "type": task_type,
                    "text": self.text,
                    "task_metadata": kwargs,
                }
                # If agents weren't provided, use the Tasks instance default.
                if not task_dict.get("agents"):
                    task_dict["agents"] = self.agents
                # Append this task to the Tasks chain.
                self.tasks.append(task_dict)
                return self  # Allow chaining.

            # Optionally, set the __name__ of the method to the task type.
            chainable_method.__name__ = task_type

            # Register this chainable method in the global dictionary.
            CHAINABLE_METHODS[task_type] = chainable_method

            # **Attach the chainable method directly to the Tasks class.**
            setattr(Workflow, task_type, chainable_method)

        return executor_fn

    return decorator


# used in tests
def _unregister_custom_task(task_type: str, chainable: bool = True):
    del TASK_EXECUTOR_REGISTRY[task_type]
    if chainable:
        del CHAINABLE_METHODS[task_type]
        delattr(Workflow, task_type)


# decorator to register custom workflows
def register_custom_workflow(name: str):
    def decorator(func):
        CUSTOM_WORKFLOW_REGISTRY[name] = func
        return func

    return decorator


# decorator to register tools
def register_tool(executor_fn: Callable):
    tool_name = executor_fn.__name__

    if tool_name in TOOLS_REGISTRY:
        raise ValueError(f"Tool '{tool_name}' is already registered.")

    # Register the executor function explicitly
    TOOLS_REGISTRY[tool_name] = Tool(name=tool_name, fn=executor_fn)
    logger.debug(f"Registered tool '{tool_name}' safely in TOOLS_REGISTRY.")

    return executor_fn