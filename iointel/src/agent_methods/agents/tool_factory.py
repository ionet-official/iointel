import re
import inspect
from pydantic import BaseModel
from typing import Callable, List
import textwrap

from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.io_logger import get_component_logger
from iointel.src.agent_methods.data_models.datamodels import Tool
from iointel.src.agent_methods.data_models.datamodels import AgentParams


logger = get_component_logger("TOOL_FACTORY")


def rehydrate_tool(tool_def: Tool) -> Callable:
    """
    Reconstruct a Python function from a Tool definition.
    This function compiles and executes the source code stored in tool_def.body,
    after removing any decorator lines and dedenting the source code.
    """
    if not tool_def.body:
        raise ValueError(
            f"No source code (body) available to rehydrate tool: {tool_def.name}"
        )

    # First, remove any decorator lines.
    cleaned_source = re.sub(r"^\s*@.*\n", "", tool_def.body, flags=re.MULTILINE)
    # If the cleaned source still starts with a decorator on the same line, remove it.
    cleaned_source = re.sub(r"^@\S+\s+", "", cleaned_source)

    # Now dedent the source code to remove common leading whitespace.
    dedented_source = textwrap.dedent(cleaned_source)

    logger.debug(
        f"Cleaned and dedented source for tool '{tool_def.name}':\n{dedented_source}"
    )

    try:
        code_obj = compile(
            dedented_source, filename=f"<tool {tool_def.name}>", mode="exec"
        )
    except Exception as e:
        raise ValueError(f"Error compiling source for tool {tool_def.name}: {e}")

    namespace = {}
    exec(code_obj, globals(), namespace)
    fn = namespace.get(tool_def.name)
    logger.debug(f"Rehydrated tool '{tool_def.name}' as function: {fn}")
    if fn is None or not callable(fn):
        raise ValueError(
            f"Could not rehydrate tool: function '{tool_def.name}' not found or not callable in the source code."
        )
    return fn


def instantiate_stateful_tool(tool: Tool, state_args: dict | None) -> BaseModel | None:
    if not isinstance(state_args, dict):
        return None
    if tool.fn.__qualname__.count(".") != 1:
        raise ValueError(f"Tool {tool.name} is not nested correctly")
    tool_cls: type[BaseModel] = tool.fn.__globals__[tool.fn.__qualname__.split(".")[0]]
    tool_obj = tool_cls.model_validate(state_args)
    return tool_obj


def resolve_tools(
    params: AgentParams,
    tool_instantiator: Callable[
        [Tool, dict | None], BaseModel | None
    ] = instantiate_stateful_tool,
) -> List[Tool]:
    """
    Resolve the tools in an AgentParams object.
    Each tool in params.tools is expected to be either:
      - a string - tool name,
      - a pair of (string, dict) - tool name + args to reconstruct tool self,
      - a dict (serialized Tool) with a "body" field,
      - a Tool instance,
      - or a callable.
    In the dict case, we ensure that the "body" is preserved.

    The `tool_instantiator` is called when a tool is "stateful",
    i.e. it is an instancemethod, and its `self` is not yet initialized.
    """
    resolved_tools = []
    for tool_data in params.tools:
        state_args: dict | None = None
        if isinstance(tool_data, (tuple, list)):
            logger.debug(f"Looking up registry for stateful tool: {tool_data}")
            try:
                name, state_args = tool_data
                if not isinstance(name, str) or not isinstance(state_args, dict):
                    raise ValueError("Incorrect types of pair of name and args")
            except ValueError as err:
                raise ValueError(
                    f"Stateful tool data should be a pair of name and args, got {tool_data}"
                ) from err
            tool_data = name
        if isinstance(tool_data, str):
            logger.debug(f"Looking up the registry for tool `{tool_data}`")
            # Use centralized tool resolution
            from ...utilities.tool_registry_utils import resolve_tool
            try:
                tool_obj = resolve_tool(tool_data, allow_unregistered=False)
                logger.debug(f"Resolved tool: name={tool_obj.name}, fn={tool_obj.fn.__name__}")
                resolved_tools.append(tool_obj.model_copy(update={"fn_self": tool_obj.fn_self}))
                continue
            except ValueError as e:
                raise ValueError(f"Tool {tool_data} is not known") from e
        elif isinstance(tool_data, dict):
            logger.debug(f"Rehydrating tool from dict: {tool_data}")
            tool_obj = Tool.model_validate(tool_data)
            if "body" in tool_data:
                tool_obj.body = tool_data["body"]
            # If the dict has a name, try to resolve it as a string first
            if "name" in tool_data:
                tool_name = tool_data["name"]
                logger.debug(f"Dict has tool name '{tool_name}', trying string resolution first")
                try:
                    from ...utilities.tool_registry_utils import resolve_tool
                    resolved_tool = resolve_tool(tool_name, allow_unregistered=False)
                    logger.debug(f"Successfully resolved dict tool by name: {tool_name}")
                    resolved_tools.append(resolved_tool.model_copy(update={"fn_self": resolved_tool.fn_self}))
                    continue
                except ValueError:
                    logger.debug(f"Could not resolve dict tool by name '{tool_name}', continuing with body-based resolution")
        elif isinstance(tool_data, Tool):
            logger.debug(f"Rehydrating tool from Tool instance: {tool_data}")
            tool_obj = tool_data
            if tool_obj.body is None:
                raise ValueError(
                    f"Tool instance {tool_obj.name} has no body to rehydrate."
                )
            else:
                logger.debug(f"Tool instance has body: {tool_obj.body}")
        elif callable(tool_data):
            logger.debug(f"Reusing callable tool: {tool_data}")
            tool_obj = Tool.from_function(tool_data)
        else:
            raise ValueError(
                "Unexpected type for tool_data; expected str, dict, Tool instance, or callable."
            )

        # For non-string tool lookups, use centralized tool resolution
        if not isinstance(tool_data, str):
            # Use centralized tool resolution for all tool types
            from ...utilities.tool_registry_utils import resolve_tool
            try:
                resolved_tool = resolve_tool(tool_obj, allow_unregistered=False)
                fn_name = resolved_tool.fn.__name__ if resolved_tool.fn else "None"
                logger.debug(f"Resolved non-string tool: name={resolved_tool.name}, fn={fn_name}")
                resolved_tools.append(resolved_tool.model_copy(update={"fn_self": resolved_tool.fn_self}))
                continue
            except ValueError as e:
                logger.warning(f"Could not resolve tool {tool_obj.name} using centralized resolver: {e}")
                # Fallback to old body comparison logic only if centralized resolution fails
                registered_tool_name, registered_tool = next(
                    (
                        (name, t)
                        for name, t in TOOLS_REGISTRY.items()
                        if t.body == tool_obj.body
                    ),
                    (None, None),
                )
        else:
            # This should not happen since string lookups are handled above
            registered_tool_name = tool_data
            registered_tool = TOOLS_REGISTRY.get(tool_data)
        if registered_tool_name:
            logger.debug(
                f"Tool '{tool_obj.name}' found in TOOLS_REGISTRY under the custom name '{registered_tool_name}'."
            )
            if state_args is not None and (
                not tool_obj.fn_metadata or not tool_obj.fn_metadata.stateful
            ):
                raise ValueError(
                    f"Tool {tool_obj.name} got state args but is not marked as stateful"
                )
            if (
                tool_obj.fn_metadata
                and tool_obj.fn_metadata.stateful
                and tool_obj.fn_self is None
            ):
                fn_self = tool_instantiator(tool_obj, state_args)
                if fn_self is not None:
                    fn_method = getattr(fn_self, tool_obj.fn.__name__, None)
                    if (
                        not inspect.ismethod(fn_method)
                        or fn_method.__func__ != tool_obj.fn
                    ):
                        raise ValueError(
                            f"Tool {tool_obj.name} got code replaced when intantiating, spoofing detected"
                        )
                    tool_obj = tool_obj.instantiate_from_state(fn_self)

            # Use the registered tool which should have fn populated
            resolved_tools.append(
                registered_tool.model_copy(update={"fn_self": tool_obj.fn_self if hasattr(tool_obj, 'fn_self') else None})
            )
            continue
        else:
            logger.warning(
                f"Tool '{tool_obj.name}' not found in TOOLS_REGISTRY, and rehydration is disabled for security."
            )
            continue
    return resolved_tools
