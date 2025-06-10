from typing import Any, Callable, Dict, Optional, Union
from agno.tools.e2b import E2BTools as AgnoE2BTools

from agno.agent import Agent
from agno.team import Team
from pydantic import Field

from .common import make_base, wrap_tool


class E2B(make_base(AgnoE2BTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    run_code: bool = Field(default=True, frozen=True)
    upload_file_: bool = Field(default=True, frozen=True)
    download_result: bool = Field(default=True, frozen=True)
    filesystem: bool = Field(default=False, frozen=True)
    internet_access: bool = Field(default=False, frozen=True)
    sandbox_management: bool = Field(default=False, frozen=True)
    timeout: int = Field(default=300, frozen=True)
    sandbox_options: Optional[Dict[str, Any]] = Field(default=None, frozen=True)
    command_execution: bool = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            run_code=self.run_code,
            upload_file=self.upload_file_,
            download_result=self.download_result,
            filesystem=self.filesystem,
            internet_access=self.internet_access,
            sandbox_management=self.sandbox_management,
            timeout=self.timeout,
            sandbox_options=self.sandbox_options,
            command_execution=self.command_execution,
        )

    @wrap_tool("agno_e2b_", AgnoE2BTools.duckduckgo_search)
    def run_python_code(self, code: str) -> str:
        return self.__tool.run_python_code(code)

    @wrap_tool("agno_e2b_", AgnoE2BTools.duckduckgo_search)
    def upload_file(self, file_path: str, sandbox_path: Optional[str] = None) -> str:
        return self.__tool.upload_file(file_path, sandbox_path)

    def download_png_result(
        self,
        agent: Union[Agent, Team],
        result_index: int = 0,
        output_path: Optional[str] = None,
    ) -> str:
        return self.__tool.download_png_result(agent, result_index, output_path)

    def download_chart_data(
        self,
        agent: Agent,
        result_index: int = 0,
        output_path: Optional[str] = None,
        add_as_artifact: bool = True,
    ) -> str:
        return self.__tool.download_chart_data(
            agent, result_index, output_path, add_as_artifact
        )

    def download_file_from_sandbox(
        self, sandbox_path: str, local_path: Optional[str] = None
    ) -> str:
        return self.__tool.download_file_from_sandbox(sandbox_path, local_path)

    def run_command(
        self,
        command: str,
        on_stdout: Optional[Callable] = None,
        on_stderr: Optional[Callable] = None,
        background: bool = False,
    ) -> str:
        return self.__tool.run_command(command, on_stdout, on_stderr, background)

    def stream_command(self, command: str) -> str:
        return self.__tool.stream_command(command)

    def run_background_command(self, command: str) -> Any:
        return self.__tool.run_background_command(command)

    def kill_background_command(self, command_obj: Any) -> str:
        return self.__tool.kill_background_command(command_obj)

    def list_files(self, directory_path: str = "/") -> str:
        return self.__tool.list_files(directory_path)

    def read_file_content(self, file_path: str, encoding: str = "utf-8") -> str:
        return self.__tool.read_file_content(file_path, encoding)

    def write_file_content(self, file_path: str, content: str) -> str:
        return self.__tool.write_file_content(file_path, content)

    def watch_directory(self, directory_path: str, duration_seconds: int = 5) -> str:
        return self._tool.watch_directory(directory_path, duration_seconds)

    def get_public_url(self, port: int) -> str:
        return self._tool.get_public_url(port)

    def run_server(self, command: str, port: int) -> str:
        return self._tool.run_server(command, port)

    def set_sandbox_timeout(self, timeout: int) -> str:
        return self._tool.set_sandbox_timeout(timeout)

    def get_sandbox_status(self) -> str:
        return self._tool.get_sandbox_status()

    def shutdown_sandbox(self) -> str:
        return self._tool.shutdown_sandbox()

    def list_running_sandboxes(self) -> str:
        return self._tool.list_running_sandboxes()
