from ..data_models.datamodels import AgentParams
from typing import Callable, List
import re
from ...utilities.registries import TOOLS_REGISTRY
from ...utilities.helpers import make_logger
from ..data_models.datamodels import Tool

import textwrap


logger = make_logger(__name__)


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


def resolve_tools(params: AgentParams) -> List:
    """
    Resolve the tools in an AgentParams object.
    Each tool in params.tools is expected to be either:
      - a dict (serialized Tool) with a "body" field,
      - a Tool instance,
      - or a callable.
    In the dict case, we ensure that the "body" is preserved.
    """
    resolved_tools = []
    for tool_data in params.tools:
        if isinstance(tool_data, dict):
            logger.debug(f"Rehydrating tool from dict: {tool_data}")
            tool_obj = Tool.model_validate(tool_data)
            if "body" in tool_data:
                tool_obj.body = tool_data["body"]
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
                "Unexpected type for tool_data; expected dict, Tool instance, or callable."
            )

        registered_tool_name, registered_tool = next(
            (
                (name, t)
                for name, t in TOOLS_REGISTRY.items()
                if t.body == tool_obj.body
            ),
            (None, None),
        )
        if registered_tool_name:
            logger.debug(
                f"Tool '{tool_obj.name}' found in TOOLS_REGISTRY under the custom name '{registered_tool_name}'."
            )
            resolved_tools.append(registered_tool)
            continue
        else:
            logger.warning(
                f"Tool '{tool_obj.name}' not found in TOOLS_REGISTRY, and rehydration is disabled for security."
            )
            continue
    return resolved_tools
