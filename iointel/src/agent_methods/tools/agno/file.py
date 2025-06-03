from pathlib import Path
from typing import Optional
from functools import wraps
from agno.tools.file import FileTools as AgnoFileTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class File(DisableAgnoRegistryMixin, AgnoFileTools):
    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__(base_dir=base_dir)

    @register_tool(name="file_read")
    @wraps(AgnoFileTools.read_file)
    def read_file(self, file_name: str) -> str:
        return super().read_file(file_name)

    @register_tool(name="file_list")
    @wraps(AgnoFileTools.list_files)
    def list_files(self) -> str:
        return super().list_files()

    @register_tool(name="file_save")
    @wraps(AgnoFileTools.save_file)
    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        return super().save_file(contents, file_name, overwrite)
