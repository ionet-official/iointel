from typing import Optional
from agno.tools.python import PythonTools as AgnoPythonTools

from .common import make_base, wrap_tool


class Python(make_base(AgnoPythonTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("run_shell_command", AgnoPythonTools.run_shell_command)
    def run_shell_command(self, args: list[str], tail: int = 100) -> str:
        return self._tool.run_shell_command(args, tail)

    @wrap_tool("save_to_file_and_run", AgnoPythonTools.save_to_file_and_run)
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
        "run_python_file_return_variable",
        AgnoPythonTools.run_python_file_return_variable,
    )
    def run_python_file_return_variable(
        self, file_name: str, variable_to_return: Optional[str] = None
    ) -> str:
        return self._tool.run_python_file_return_variable(file_name, variable_to_return)

    def read_file(self, file_name: str) -> str:
        return self._tool.read_file(file_name)

    def list_files(self) -> str:
        return self._tool.list_files()

    def run_python_code(
        self, code: str, variable_to_return: Optional[str] = None
    ) -> str:
        return self._tool.run_python_code(code, variable_to_return)

    def pip_install_package(self, package_name: str) -> str:
        return self._tool.pip_install_package(package_name)

    def uv_pip_install_package(self, package_name: str) -> str:
        return self._tool.uv_pip_install_package(package_name)
