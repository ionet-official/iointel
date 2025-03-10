from functools import wraps
from typing import Callable, Optional
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

    def decorator(tool_fn: Callable):
        # Register the executor function for later task execution.
        prefect_task = task(tool_fn, name=task_type, persist_result=False)
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

        return tool_fn

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


def register_tool(_fn=None, *, name: Optional[str] = None):
    def decorator(executor_fn: Callable):
        tool_name = name or executor_fn.__name__

        if tool_name in TOOLS_REGISTRY:
            existing_tool = TOOLS_REGISTRY[tool_name]
            if executor_fn.__code__.co_code != existing_tool.fn.__code__.co_code:
                raise ValueError(
                    f"Tool name '{tool_name}' already registered with a different function. Potential spoofing detected."
                )
            logger.debug(f"Tool '{tool_name}' is already safely registered.")
            return executor_fn

        if 'self' in executor_fn.__code__.co_varnames:
            @wraps(executor_fn)
            def wrapper(self, *args, **kwargs):
                return executor_fn(self, *args, **kwargs)

            TOOLS_REGISTRY[tool_name] = Tool.from_function(wrapper)
            logger.debug(f"Registered method tool '{tool_name}' safely.")
            return wrapper

        TOOLS_REGISTRY[tool_name] = Tool.from_function(executor_fn)
        logger.debug(f"Registered tool '{tool_name}' safely.")
        return executor_fn

    if callable(_fn):
        return decorator(_fn)
