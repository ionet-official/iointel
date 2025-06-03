import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from iointel.src.agent_methods.tools.agno.file import File


class TestFile:
    @pytest.fixture
    def temp_dir(self):
        with TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def file_tool(self, temp_dir):
        return File(base_dir=temp_dir)

    def test_read_write_file(self, file_tool, temp_dir):
        test_content = "Hello, World!"
        test_file_name = "test.txt"

        # Write file
        result = file_tool.save_file(test_content, test_file_name)
        assert result == test_file_name

        # Read file
        content = file_tool.read_file(test_file_name)
        assert content == test_content

    def test_list_files(self, file_tool, temp_dir):
        # Create some test files
        test_files = ["test1.txt", "test2.txt", "test3.py"]
        for file in test_files:
            file_tool.save_file("test content", file)

        # List all files
        files = file_tool.list_files()
        assert all(file in files for file in test_files)

    def test_file_not_found(self, file_tool):
        r = file_tool.read_file("nonexistent.txt")
        assert r.startswith("Error reading file:"), "File is supposed to not be found"

    def test_invalid_path(self, file_tool):
        # Test with invalid path
        with pytest.raises(Exception):
            file_tool.write_file("/invalid/path/test.txt", "content")
