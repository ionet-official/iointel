import pytest
from pydantic import SecretStr

from iointel.src.agents import Agent
from pydantic_ai.models.openai import OpenAIModel


@pytest.mark.parametrize("prefix", ["OPENAI_API", "IO_API"])
def test_agent_default_model(prefix, monkeypatch):
    """
    Test that Agent uses OpenAIModel with environment variables by default.
    """
    monkeypatch.setenv(f"{prefix}_KEY", "fake_api_key")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "http://fake-url.com")

    a = Agent(
        name="TestAgent",
        instructions="You are a test agent.",
    )
    assert isinstance(a.model, OpenAIModel), (
        "Agent should default to ChatOpenAI if no provider is specified."
    )
    assert a.name == "TestAgent"
    assert "test agent" in a.instructions.lower()


def test_agent_api_key_from_params():
    """
    Test that Agent correctly passes api_key and base_url params.
    """
    api_key = "fake_api_key"
    base_url = "http://fake-url.com"

    a = Agent(
        name="TestAgent",
        instructions="You are a test agent.",
        api_key=api_key,
        base_url=base_url,
    )
    assert isinstance(a.api_key, SecretStr), "Api key should be stored as SecretStr."
    assert a.api_key.get_secret_value() == api_key, (
        "Api key value should be taken from params."
    )
    assert a.model.base_url == base_url, "Base url value should be taken from params."


def test_agent_run():
    """
    Basic check that the agent's run method calls Agent.run under the hood.
    We'll mock it or just ensure it doesn't crash.
    """
    a = Agent(name="RunAgent", instructions="Test run method.")
    # Because there's no real LLM here (mock credentials), the actual run might fail or stub.
    # We can call run with a stub prompt and see if it returns something or raises a specific error.
    result = a.run("Hello world")
    assert result is not None, "Expected a result from the agent run."
    # with pytest.raises(Exception):
    #    # This might raise an error due to fake API key or no actual LLM.
    #    a.run("Hello world")
