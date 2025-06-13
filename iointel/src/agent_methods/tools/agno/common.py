from pydantic import BaseModel
from functools import wraps
from agno.tools import Toolkit

from ..utils import register_tool


class DisableAgnoRegistryMixin:
    """
    Put this as first parent class when inheriting
    from Agno tool to disable Agno registry,
    because we only care about our own registry."""

    def _register_tools(self):
        """Disabled in favour of iointel registry."""

    def register(self, function, name=None):
        """Disabled in favour of iointel registry."""


def make_base(agno_tool_cls: type[Toolkit]):
    class BaseAgnoTool(BaseModel):
        class Inner(DisableAgnoRegistryMixin, agno_tool_cls):
            pass

        def _get_tool(self) -> Inner:
            raise NotImplementedError()

        _tool: Inner

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self._tool = self._get_tool()

    return BaseAgnoTool


def wrap_tool(name, agno_method):
    def wrapper(func):
        return register_tool(name=name)(wraps(agno_method)(func))

    return wrapper
