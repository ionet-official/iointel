from typing import Optional
from agno.tools.local_file_system import (
    LocalFileSystemTools as AgnoLocalFileSystemTools,
)
from .common import make_base, wrap_tool
from pydantic import Field


class LocalFileSystem(make_base(AgnoLocalFileSystemTools)):
    target_directory: Optional[str] = Field(default=None, frozen=True)
    default_extension: str = Field(default="txt", frozen=True)

    def _get_tool(self):
        return self.Inner(
            target_directory=self.target_director_,
            default_extension=self.default_extension,
        )

    @wrap_tool("agno__localfilesystem__write_file", AgnoLocalFileSystemTools.write_file)
    def write_file(
        self,
        content: str,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> str:
        return self.write_file(content, filename, directory, extension)

    @wrap_tool("agno__localfilesystem__read_file", AgnoLocalFileSystemTools.read_file)
    def read_file(self, filename: str, directory: Optional[str] = None) -> str:
        return self.read_file(filename, directory)
