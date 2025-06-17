from pathlib import Path
from typing import Optional
from agno.tools.python import PythonTools as AgnoPythonTools
from pydantic import Field
from .common import make_base, wrap_tool


class Python(make_base(AgnoPythonTools)):
    base_dir: Optional[Path] = Field(default=None, frozen=True)
    save_and_run: bool = Field(default=True, frozen=True)
    pip_install: bool = Field(default=False, frozen=True)
    uv_pip_install: bool = Field(default=False, frozen=True)
    run_code: bool = Field(default=False, frozen=True)
    list_files_: bool = Field(default=False, frozen=True)
    run_files: bool = Field(default=False, frozen=True)
    safe_globals: Optional[dict] = Field(default=None, frozen=True)
    safe_locals: Optional[dict] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            base_dir=self.base_dir,
            save_and_run=self.save_and_run,
            pip_install=self.pip_install,
            uv_pip_install=self.uv_pip_install,
            run_code=self.run_code,
            list_files=self.list_files_,
            run_files=self.run_files,
            read_files=True,
            safe_globals=self.safe_globals,
            safe_locals=self.safe_locals,
        )

    @wrap_tool("agno__python__run_shell_command", AgnoPythonTools.run_shell_command)
    def run_shell_command(self, args: list[str], tail: int = 100) -> str:
        return self._tool.run_shell_command(args, tail)

    @wrap_tool(
        "agno__python__save_to_file_and_run", AgnoPythonTools.save_to_file_and_run
    )
    def save_to_file_and_run(
        self,
        file_name: str,
        code: str,
        variable_to_return: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        return self._tool.save_to_file_and_run(
            file_name, code, variable_to_return, overwrite
        )

    @wrap_tool(
        "agno__python__run_python_file_return_variable",
        AgnoPythonTools.run_python_file_return_variable,
    )
    def run_python_file_return_variable(
        self, file_name: str, variable_to_return: Optional[str] = None
    ) -> str:
        return self._tool.run_python_file_return_variable(file_name, variable_to_return)

    @wrap_tool("agno__python__read_file", AgnoPythonTools.read_file)
    def read_file(self, file_name: str) -> str:
        return self._tool.read_file(file_name)

    @wrap_tool("agno__python__list_files", AgnoPythonTools.list_files)
    def list_files(self) -> str:
        return self._tool.list_files()

    @wrap_tool("agno__python__run_python_code", AgnoPythonTools.run_python_code)
    def run_python_code(
        self, code: str, variable_to_return: Optional[str] = None
    ) -> str:
        return self._tool.run_python_code(code, variable_to_return)

    @wrap_tool("agno__python__pip_install_package", AgnoPythonTools.pip_install_package)
    def pip_install_package(self, package_name: str) -> str:
        return self._tool.pip_install_package(package_name)

    @wrap_tool(
        "agno__python__uv_pip_install_package", AgnoPythonTools.uv_pip_install_package
    )
    def uv_pip_install_package(self, package_name: str) -> str:
        return self._tool.uv_pip_install_package(package_name)
