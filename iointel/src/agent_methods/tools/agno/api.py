from typing import Any, Dict, Literal, Optional
from agno.tools.api import CustomApiTools as AgnoCustomApiTools

from .common import make_base, wrap_tool


class Api(make_base(AgnoCustomApiTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("make_request", AgnoCustomApiTools.make_request)
    def make_request(
        self,
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._tool.make_request()
