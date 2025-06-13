import tempfile
import pytest
from iointel.src.agent_methods.tools.agno.shell import Shell


@pytest.fixture
def shell_tool():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Shell(base_dir=temp_dir)


def test_shell_execute_command(shell_tool):
    result = shell_tool.run_shell_command("echo 'hello world'")
    assert "hello world" in result
