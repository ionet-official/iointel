from typing import Any, Callable, Optional, List
from agno.tools.toolkit import Toolkit as AgnoToolkit
from .common import make_base, wrap_tool


class To(make_base(AgnoToolkit)):
    def _get_tool(self):
        return self.Inner(
            name=self.name_,
            tools=self.tools_,
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
            include_tools=self.include_tools_,
            exclude_tools=self.exclude_tools_,
            requires_confirmation_tools=self.requires_confirmation_tools_,
            external_execution_required_tools=self.external_execution_required_tools_,
            stop_after_tool_call_tools=self.stop_after_tool_call_tools_,
            show_result_tools=self.show_result_tools_,
            cache_results=self.cache_results_,
            cache_ttl=self.cache_ttl_,
            cache_dir=self.cache_dir_,
            auto_register=self.auto_register_,
        )

    @wrap_tool("agno__to___check_tools_filters", AgnoToolkit._check_tools_filters)
    def _check_tools_filters(
        self,
        available_tools: List[str],
        include_tools: Optional[list[str]] = None,
        exclude_tools: Optional[list[str]] = None,
    ) -> None:
        return self._check_tools_filters(
            self, available_tools, include_tools, exclude_tools
        )

    @wrap_tool("agno__to___register_tools", AgnoToolkit._register_tools)
    def _register_tools(self) -> None:
        return self._register_tools(self)

    @wrap_tool("agno__to__register", AgnoToolkit.register)
    def register(self, function: Callable[..., Any], name: Optional[str] = None):
        return self.register(self, function, name)
