import pytest

from datetime import datetime
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
    toolcall_happened = None

    def get_current_datetime() -> str:
        nonlocal toolcall_happened
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        toolcall_happened = {
            'datetime': current_datetime,
        }
        return current_datetime

    a = Agent(name="RunAgent", instructions="Return current datetime, provided by the tool.",
              tools=[get_current_datetime])
    result = a.run("Return current datetime. Use the tools provided")
    assert result is not None, "Expected a result from the agent run."
    assert toolcall_happened is not None, "Make sure the tool was actually called"
    assert toolcall_happened['datetime'] in result, "Make sure the result of the toolcall matches the return value of the agent"

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

    a = Agent(name="RunAgent", instructions="Add two numbers.", tools=[add_two_numbers])
    result = a.run("Add numbers 2 and 7. Use the tools provided")
    assert result is not None, "Expected a result from the agent run."
    assert '9' in result, "Result should be 9"
    assert toolcall_happened is not None, "Make sure the tool was actually called"
    assert str(toolcall_happened['a+b']) == result, "Make sure the result of the toolcall matches the return value of the agent"
