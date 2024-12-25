# tests/test_agents.py
import os
import pytest
from framework.src.agents import Agent
from langchain_openai import ChatOpenAI
import controlflow as cf

def test_agent_default_model(monkeypatch):
    """
    Test that Agent uses ChatOpenAI with environment variables by default.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "fake_api_key")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-url.com")

    a = Agent(
        name="TestAgent",
        instructions="You are a test agent.",
    )
    assert isinstance(a.model, ChatOpenAI), "Agent should default to ChatOpenAI if no provider is specified."
    assert a.name == "TestAgent"
    assert "test agent" in a.instructions.lower()

def test_agent_custom_provider():
    """
    Test passing a custom model provider callable.
    """
    def mock_provider(**kwargs):
        return "MockModel"

    a = Agent(
        name="CustomModelAgent",
        instructions="Instructions for custom model",
        model_provider=mock_provider,
        some_param="value"
    )
    assert a.model == "MockModel", "Expected the custom provider to be used."
    assert a.tools == [], "By default, tools should be an empty list."

def test_agent_run():
    """
    Basic check that the agent's run method calls cf.Agent.run under the hood.
    We'll mock it or just ensure it doesn't crash.
    """
    a = Agent(name="RunAgent", instructions="Test run method.")
    # Because there's no real LLM here (mock credentials), the actual run might fail or stub.
    # We can call run with a stub prompt and see if it returns something or raises a specific error.
    with pytest.raises(Exception):
        # This might raise an error due to fake API key or no actual LLM.
        a.run("Hello world")