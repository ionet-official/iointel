from typing import List, Optional
from functools import wraps
from agno.tools.shell import ShellTools as AgnoShellTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class Shell(DisableAgnoRegistryMixin, AgnoShellTools):
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(base_dir=base_dir)

    @register_tool(name="run_shell_command")
    @wraps(AgnoShellTools.run_shell_command)
    def run_shell_command(self, args: List[str], tail: int = 100) -> str:
        return super().run_shell_command(args, tail)
