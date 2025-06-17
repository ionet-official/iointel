from typing import Any, Dict, Literal, Optional
from agno.tools.api import CustomApiTools as AgnoCustomApiTools
from pydantic import Field

from .common import make_base, wrap_tool


class Api(make_base(AgnoCustomApiTools)):
    base_url: Optional[str] = Field(default=None, frozen=True)
    username: Optional[str] = Field(default=None, frozen=True)
    password: Optional[str] = Field(default=None, frozen=True)
    api_key: Optional[str] = Field(default=None, frozen=True)
    headers: Optional[Dict[str, str]] = Field(default=None, frozen=True)
    verify_ssl: bool = Field(default=True, frozen=True)
    timeout: int = Field(default=30, frozen=True)

    def _get_tool(self):
        return self.Inner(
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            api_key=self.api_key,
            headers=self.headers,
            verify_ssl=self.verify_ssl,
            timeout=self.timeout,
            make_request=True,
        )

    @wrap_tool("agno__api__make_request", AgnoCustomApiTools.make_request)
    def make_request(
        self,
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._tool.make_request(
            endpoint=self.endpoint,
            method=self.method,
            params=self.params,
            data=self.data,
            headers=self.headers,
            json_data=self.json_data,
        )
