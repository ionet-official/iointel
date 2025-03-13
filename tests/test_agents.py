import pytest
from unittest.mock import patch, MagicMock
from iointel.src.agents import Agent
from pydantic_ai.models.openai import OpenAIModel


@pytest.mark.parametrize("prefix", ["OPENAI_API", "IO_API"])
def test_agent_default_model(prefix, monkeypatch):
    """
    Test that Agent uses OpenAIModel with environment variables by default.
    """
    monkeypatch.setenv(f"{prefix}_KEY", "fake_api_key")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "http://fake-url.com")

    with patch("iointel.src.agents.OpenAIModel") as MockModel:
        mock_instance = MagicMock(spec=OpenAIModel)
        MockModel.return_value = mock_instance

        a = Agent(
            name="TestAgent",
            instructions="You are a test agent.",
        )

        MockModel.assert_called_once_with(
            api_key="fake_api_key",
            base_url="http://fake-url.com",
            model="gpt-4o-mini",
        )

        assert isinstance(a.model, OpenAIModel), (
            "Agent should default to OpenAIModel if no provider is specified."
        )
        assert a.name == "TestAgent"
        assert "test agent" in a.instructions.lower()

def test_agent_run():
    """
    Basic check that the agent's run method calls Agent.run under the hood.
    """
    with patch.object(Agent, 'run', return_value="Mocked response") as mock_run:
        a = Agent(name="RunAgent", instructions="Test run method.")

        result = a.run("Hello world")

        mock_run.assert_called_once_with("Hello world")
        assert result == "Mocked response", "Expected mocked result from agent run."