from pathlib import Path
from agno.tools.file import FileTools as AgnoFileTools

from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool


class File(make_base(AgnoFileTools)):
    base_dir: Path | None = None

    def _get_tool(self):
        return self.Inner(base_dir=self.base_dir)

    @wrap_tool("file_read", AgnoFileTools.read_file)
    def read_file(self, file_name: str) -> str:
        return self._tool.read_file(file_name)

    @wrap_tool("file_list", AgnoFileTools.list_files)
    def list_files(self) -> str:
        return self._tool.list_files()

    @wrap_tool("file_save", AgnoFileTools.save_file)
    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        return self._tool.save_file(contents, file_name, overwrite)
