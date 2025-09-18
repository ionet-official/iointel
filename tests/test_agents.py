import pytest
from pydantic import SecretStr
from unittest.mock import patch

from iointel.src.agents import Agent, StreamableAgentResult
from iointel.src.agent_methods.data_models.datamodels import AgentResult
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


@pytest.mark.asyncio
async def test_run_stream_blocking_mode():
    """
    Test 1: run_stream works in blocking mode (backward compatibility)
    """
    agent = Agent(name="StreamAgent", instructions="Test streaming.")

    # Mock the _stream_tokens method to return predictable data
    mock_agent_result = AgentResult(
        result="Hello, World!",
        conversation_id="test-123",
        full_result=None,
        tool_usage_results=[],
    )

    async def mock_stream_tokens(*args, **kwargs):
        # New _stream_tokens yields individual deltas
        yield "Hello"  # Individual delta
        yield ", "  # Individual delta
        yield "World!"  # Individual delta
        yield {
            "__final__": True,
            "content": "Hello, World!",  # Full accumulated content
            "agent_result": mock_agent_result.full_result,
        }

    with patch.object(agent, "_stream_tokens", mock_stream_tokens):
        with patch.object(
            agent, "_postprocess_agent_result", return_value=mock_agent_result
        ):
            # Test blocking mode - should work like before
            streamable_result = agent.run_stream("Test query")
            assert isinstance(streamable_result, StreamableAgentResult)

            # When awaited, should return AgentResult (backward compatibility)
            result = await streamable_result
            assert isinstance(result, AgentResult)
            assert result.result == "Hello, World!"
            assert result.conversation_id == "test-123"


@pytest.mark.asyncio
async def test_run_stream_streaming_mode():
    """
    Test 2: run_stream works in streaming mode (new functionality)
    """
    agent = Agent(name="StreamAgent", instructions="Test streaming.")

    mock_agent_result = AgentResult(
        result="Hello, World!",
        conversation_id="test-123",
        full_result=None,
        tool_usage_results=[],
    )

    async def mock_stream_tokens(*args, **kwargs):
        # New _stream_tokens yields individual deltas
        yield "Hello"  # Individual delta
        yield ", "  # Individual delta
        yield "World!"  # Individual delta
        yield {
            "__final__": True,
            "content": "Hello, World!",  # Full accumulated content
            "agent_result": mock_agent_result.full_result,
        }

    with patch.object(agent, "_stream_tokens", mock_stream_tokens):
        with patch.object(
            agent, "_postprocess_agent_result", return_value=mock_agent_result
        ):
            # Test streaming mode - should yield tokens then final result
            stream_result = await agent.run_stream("Test query")
            assert isinstance(stream_result, StreamableAgentResult)

            tokens = []
            final_result = None

            async for chunk in stream_result:
                if isinstance(chunk, str):
                    tokens.append(chunk)
                else:
                    final_result = chunk

            # Now expecting individual deltas
            assert tokens == ["Hello", ", ", "World!"]
            assert isinstance(final_result, AgentResult)
            assert final_result.result == "Hello, World!"


@pytest.mark.asyncio
async def test_streamable_agent_result_single_consumption():
    """
    Test 3: StreamableAgentResult can only be consumed once
    """

    async def mock_generator():
        yield "token1"
        yield "token2"
        yield AgentResult(
            result="final",
            conversation_id="test",
            full_result=None,
            tool_usage_results=[],
        )

    streamable = StreamableAgentResult(mock_generator())

    # First consumption should work
    tokens = []
    async for chunk in streamable:
        tokens.append(chunk)

    assert len(tokens) == 3
    assert tokens[0] == "token1"
    assert tokens[1] == "token2"
    assert isinstance(tokens[2], AgentResult)

    # Second consumption should raise error
    with pytest.raises(RuntimeError, match="Stream can only be consumed once"):
        async for chunk in streamable:
            pass
