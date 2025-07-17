from typing import List
from agno.tools.shell import ShellTools as AgnoShellTools

from .common import make_base, wrap_tool


class Shell(make_base(AgnoShellTools)):
    base_dir: str | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("run_shell_command", AgnoShellTools.run_shell_command)
    def run_shell_command(self, args: List[str], tail: int = 100) -> str:
        return self._tool.run_shell_command(args, tail)
