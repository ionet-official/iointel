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

        # Check if the tool is already in the registry.
        if tool_obj.name in TOOLS_REGISTRY:
            logger.debug(f"Tool '{tool_obj.name}' found in TOOLS_REGISTRY.")
            candidate = TOOLS_REGISTRY[tool_obj.name]
            if callable(candidate) and hasattr(candidate, "body") and candidate.body:
                resolved_tools.append(candidate)
                continue
            else:
                logger.debug(f"Rehydrating tool '{tool_obj.name}' from registry.")
                rehydrated_fn = rehydrate_tool(tool_obj)
                if not callable(rehydrated_fn):
                    raise ValueError(
                        f"Rehydrated tool for {tool_obj.name} is not callable!"
                    )
                # Create a new Tool instance that keeps the original body
                new_tool = tool_obj.model_copy(update={"fn": rehydrated_fn})
                TOOLS_REGISTRY[tool_obj.name] = new_tool
                resolved_tools.append(new_tool)
                continue
        else:
            logger.debug(
                f"Rehydrating tool '{tool_obj.name}' not found in TOOLS_REGISTRY."
            )
            rehydrated_fn = rehydrate_tool(tool_obj)
            if not callable(rehydrated_fn):
                raise ValueError(
                    f"Rehydrated tool for {tool_obj.name} is not callable!"
                )
            # Instead of model_copy(), explicitly copy the dictionary and update fn.
            original_data = tool_obj.model_dump()
            if "body" not in original_data or not original_data["body"]:
                original_data["body"] = tool_obj.body
            original_data["fn"] = rehydrated_fn
            new_tool = Tool(**original_data)
            TOOLS_REGISTRY[tool_obj.name] = new_tool
            logger.debug(
                f"Registered rehydrated tool '{tool_obj.name}' in TOOLS_REGISTRY with body: {new_tool.body}"
            )
            resolved_tools.append(new_tool)
    return resolved_tools
