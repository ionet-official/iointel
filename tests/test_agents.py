import pytest

from iointel.src.agent_methods.tools.tools import get_current_datetime
from iointel.src.agents import Agent
from langchain_openai import ChatOpenAI


@pytest.mark.parametrize("prefix", ["OPENAI_API", "IO_API"])
def test_agent_default_model(prefix, monkeypatch):
    """
    Test that Agent uses ChatOpenAI with environment variables by default.
    """
    monkeypatch.setenv(f"{prefix}_KEY", "fake_api_key")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "http://fake-url.com")

    a = Agent(
        name="TestAgent",
        instructions="You are a test agent.",
    )
    assert isinstance(
        a.model, ChatOpenAI
    ), "Agent should default to ChatOpenAI if no provider is specified."
    assert a.name == "TestAgent"
    assert "test agent" in a.instructions.lower()


def test_agent_run():
    """
    Basic check that the agent's run method calls Agent.run under the hood.
    We'll mock it or just ensure it doesn't crash.
    """
    a = Agent(
        name="RunAgent",
        instructions="Test run method.",
    )
    result = a.run("Hello world")
    assert result is not None, "Expected a result from the agent run."

def test_tool_call():
    """
    Basic check that the agent's toolcall mechanism is working
    """
    a = Agent(name="RunAgent", instructions="Test run method.", tools=[get_current_datetime])
    result = a.run("Return current datetime")
    current_date = get_current_datetime().split(' ')[0]
    assert result is not None, "Expected a result from the agent run."
    assert current_date in result, "Date should be extracted correctly"

def test_tool_call_with_addition():
    """
    Check addition tool.
    Make sure the types are resolved correctly
    """
    toolcall_happened = None

    def add_two_numbers(a: int, b: int) -> int:
        """
        Return the result of addition
        """
        nonlocal toolcall_happened
        toolcall_happened = {
            'a': a,
            'b': b,
            'a+b': a + b
        }
        return a + b

    a = Agent(name="RunAgent", instructions="Test run method.", tools=[add_two_numbers])
    result = a.run("Add numbers 2 and 7. Use the provided tool. Return result provided by the tool")
    assert result is not None, "Expected a result from the agent run."
    assert '9' in result, "Result should be 9"
    assert toolcall_happened is not None, "Make sure the tool was actually called"
    assert toolcall_happened['a+b'] == 9, "Make sure the result of the toolcall matches the return value of the agent"
