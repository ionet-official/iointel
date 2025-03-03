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