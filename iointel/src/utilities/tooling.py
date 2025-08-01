from pathlib import Path
import os
import warnings
from pydantic import BaseModel
from typing import TYPE_CHECKING
from dotenv import dotenv_values

import iointel.src.agent_methods.tools as tools_root
from iointel.src.utilities.registries import TOOLS_REGISTRY, STATEFUL_TOOL_DEFAULTS

if TYPE_CHECKING:
    from iointel.src.agent_methods.data_models.datamodels import Tool


def _find_all_tools():
    tools_dir = Path(tools_root.__file__).parent
    result: list[str] = []
    for fn in tools_dir.glob("**/*.py"):
        if fn.name != "__init__.py":
            result.append(
                ".".join(
                    [tools_root.__name__]
                    + [
                        part.removesuffix(".py")
                        for part in fn.relative_to(tools_dir).parts
                    ]
                )
            )
    return result


_TOOLS_LOADED: list[str] = []


def discover_all_tools():
    global _TOOLS_LOADED
    if _TOOLS_LOADED:
        return _TOOLS_LOADED

    result = []
    for pkg in _find_all_tools():
        existing_tools = set(TOOLS_REGISTRY)
        try:
            __import__(pkg)
        except Exception as err:
            # It could happen that importing a tool module would succeed *partially*,
            # i.e. it would register tool functions, but then something happens like
            # Pydantic refusing to create a class. We need to remove such broken
            # tools from the registry.
            broken_tools = set(TOOLS_REGISTRY) - existing_tools
            for tool in broken_tools:
                TOOLS_REGISTRY.pop(tool)
            if isinstance(err, ImportError):
                warnings.warn(f"Tool package {pkg} not available: {err}")
            else:
                warnings.warn(f"Tool package {pkg} broken: {err}")
        else:
            result.append(pkg)
    _TOOLS_LOADED = result
    return result


def get_tool_model_class(tool: "Tool"):
    if tool.fn.__qualname__.count(".") != 1:
        raise ValueError(f"Tool {tool.name} is not nested correctly")
    func_name = tool.fn.__code__.co_name
    tool_cls: type[BaseModel] | None = tool.fn.__globals__.get(
        tool.fn.__qualname__.split(".")[0]
    )
    if tool_cls is None:
        # @wraps() tool or some other edge case, try all func globals until it fits
        for obj in tool.fn.__globals__.values():
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                if (candidate := getattr(obj, func_name, None)) is not None:
                    if getattr(candidate, "__code__", None) == tool.fn.__code__:
                        tool_cls = obj
                        break
        else:
            raise ValueError(f"Cannot find {tool.name} class")
    return tool_cls


def _fill_toolcls_defaults(tool_cls: type[BaseModel], mapping: dict) -> dict:
    args = {}
    for name in tool_cls.model_fields:
        value = mapping.get(f"{tool_cls.__name__.upper()}_{name.upper()}")
        if value is not None:
            args[name] = value
    return args


def _show_toolcls_defaults(tool_cls: type[BaseModel]) -> list[str]:
    return [
        f"{tool_cls.__name__.upper()}_{name.upper()}" for name in tool_cls.model_fields
    ]


def fill_tool_defaults(mapping: str | Path | dict | None = None):
    """
    Discovers all operational tools and tries to construct default arguments
    for stateful tools by parsing the given mapping.

    If mapping is a string, it's interpreted as .env path.
    If mapping is omitted, os.environ is used.
    """
    discover_all_tools()
    if mapping is None:
        mapping = dict(os.environ)
    elif not isinstance(mapping, dict):
        mapping = dotenv_values(str(mapping))
    seen: dict[str, dict] = {}
    for tool in TOOLS_REGISTRY.values():
        if tool.fn_metadata.stateful:
            try:
                tool_cls = get_tool_model_class(tool)
            except ValueError as err:
                warnings.warn(f"Tool {tool.name} is broken: {err}")
                continue
            if (args := seen.get(str(tool_cls))) is None:
                args = seen[str(tool_cls)] = _fill_toolcls_defaults(tool_cls, mapping)
            STATEFUL_TOOL_DEFAULTS[tool.name] = args


def show_tool_default_args() -> dict[str, list[str]]:
    """
    Returns a mapping from stateful tool *class name* to
    the list of arg names as parsed from environment.
    """
    discover_all_tools()
    seen: dict[str, list[str]] = {}
    for tool in TOOLS_REGISTRY.values():
        if tool.fn_metadata.stateful:
            try:
                tool_cls = get_tool_model_class(tool)
            except ValueError as err:
                warnings.warn(f"Tool {tool.name} is broken: {err}")
                continue
            if str(tool_cls) not in seen:
                seen[str(tool_cls)] = _show_toolcls_defaults(tool_cls)
    return seen
