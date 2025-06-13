from typing import Callable, Optional


from .registries import (
    CHAINABLE_METHODS,
    TASK_EXECUTOR_REGISTRY,
    CUSTOM_WORKFLOW_REGISTRY,
    TOOLS_REGISTRY,
)
from ..workflow import Workflow
from ..agent_methods.data_models.datamodels import Tool, check_fn_name, compute_fn_name
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
        TASK_EXECUTOR_REGISTRY[task_type] = tool_fn

        if chainable:

            def chainable_method(self, **kwargs):
                # Create a task dictionary for this custom task.
                task_dict = {"type": task_type, "objective": self.objective}
                # Merge in any extra parameters passed to the chainable method.
                task_dict.update(kwargs)
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
def _unregister_custom_task(task_type: str):
    del TASK_EXECUTOR_REGISTRY[task_type]
    if task_type in CHAINABLE_METHODS:
        del CHAINABLE_METHODS[task_type]
        delattr(Workflow, task_type)


# decorator to register custom workflows
def register_custom_workflow(name: str):
    def decorator(func):
        CUSTOM_WORKFLOW_REGISTRY[name] = func
        return func

    return decorator


# decorator to register tools
def register_tool(_fn=None, name: Optional[str] = None):
    """
    Decorator that registers a tool function with the given name. If the name is not provided, the function name is used.
    Can be used as a decorator or as a function. If used as a function, the name must be provided.
    Can be used to register a method as a tool by passing the method as an argument.
    Or can be used to register a function as a tool by using it as a decorator.

    param _fn: The function to register as a tool.
    param name: The name to register the tool with. If not provided, the function name is used.
    return: The registered function or method.
    """

    def decorator(executor_fn: Callable):
        tool_name = check_fn_name(name) or compute_fn_name(executor_fn)
        if executor_fn.__qualname__.count(".") > 1:
            logger.warning(
                f"Tool name {tool_name} is too deeply nested: qualified name {executor_fn.__qualname__}"
            )

        if tool_name in TOOLS_REGISTRY:
            existing_tool = TOOLS_REGISTRY[tool_name]
            if executor_fn.__code__.co_code != existing_tool.fn.__code__.co_code:
                raise ValueError(
                    f"Tool name '{tool_name}' already registered with a different function. Potential spoofing detected."
                )
            logger.debug(f"Tool '{tool_name}' is already safely registered.")
            return executor_fn

        TOOLS_REGISTRY[tool_name] = Tool.from_function(executor_fn, name=tool_name)
        logger.debug(f"Registered tool '{tool_name}' safely.")
        return executor_fn

    if callable(_fn):
        return decorator(_fn)

    if isinstance(_fn, str):
        # Handle case @register_tool("tool_name")
        return register_tool(name=_fn)

    if _fn is not None:
        raise ValueError(
            "Invalid usage of register_tool. Must provide a callable or use name='...'."
        )

    return decorator
