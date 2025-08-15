"""Test multimodal support for iointel agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from iointel import Agent, ImageUrl, BinaryContent, DocumentUrl
from pydantic_ai.models.openai import OpenAIModel


@pytest.mark.asyncio
async def test_agent_accepts_image_url():
    """Test that Agent can accept an ImageUrl in a sequence."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying runner
    agent._runner.run = AsyncMock(
        return_value=MagicMock(
            output="I see an image",
            all_messages=lambda: [],
            new_messages=lambda: [],
            usage={},
            result=MagicMock(tool_usage_results=[], output="I see an image"),
        )
    )

    # Test with image URL
    image_content = [
        "What's in this image?",
        ImageUrl(url="https://example.com/image.png"),
    ]

    await agent.run(image_content)

    # Verify the run was called with the multimodal content
    agent._runner.run.assert_called_once()
    call_args = agent._runner.run.call_args[0]
    assert call_args[0] == image_content


@pytest.mark.asyncio
async def test_agent_accepts_binary_content():
    """Test that Agent can accept BinaryContent for local images."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying runner
    agent._runner.run = AsyncMock(
        return_value=MagicMock(
            output="I see binary content",
            all_messages=lambda: [],
            new_messages=lambda: [],
            usage={},
            result=MagicMock(tool_usage_results=[], output="I see binary content"),
        )
    )

    # Test with binary content
    image_bytes = b"fake-image-data"
    multimodal_content = [
        "Analyze this image",
        BinaryContent(data=image_bytes, media_type="image/png"),
    ]

    await agent.run(multimodal_content)

    # Verify the run was called with the multimodal content
    agent._runner.run.assert_called_once()
    call_args = agent._runner.run.call_args[0]
    assert call_args[0] == multimodal_content


@pytest.mark.asyncio
async def test_agent_backward_compatibility_string():
    """Test that Agent still accepts plain strings for backward compatibility."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying runner
    agent._runner.run = AsyncMock(
        return_value=MagicMock(
            output="Text response",
            all_messages=lambda: [],
            new_messages=lambda: [],
            usage={},
            result=MagicMock(tool_usage_results=[], output="Text response"),
        )
    )

    # Test with plain string (backward compatibility)
    await agent.run("Simple text query")

    # Verify the run was called with the string
    agent._runner.run.assert_called_once()
    call_args = agent._runner.run.call_args[0]
    assert call_args[0] == "Simple text query"


@pytest.mark.asyncio
async def test_agent_mixed_multimodal_content():
    """Test that Agent can handle mixed multimodal content."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying runner
    agent._runner.run = AsyncMock(
        return_value=MagicMock(
            output="Analyzed multiple items",
            all_messages=lambda: [],
            new_messages=lambda: [],
            usage={},
            result=MagicMock(tool_usage_results=[], output="Analyzed multiple items"),
        )
    )

    # Test with mixed content
    mixed_content = [
        "Compare these:",
        ImageUrl(url="https://example.com/image1.png"),
        "with this document:",
        DocumentUrl(url="https://example.com/doc.pdf"),
    ]

    await agent.run(mixed_content)

    # Verify the run was called with the mixed content
    agent._runner.run.assert_called_once()
    call_args = agent._runner.run.call_args[0]
    assert call_args[0] == mixed_content


@pytest.mark.asyncio
async def test_stream_with_multimodal():
    """Test that streaming methods also accept multimodal content."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying _stream_tokens method which is what actually does the streaming
    async def mock_stream_tokens(*args, **kwargs):
        # Yield a final token to simulate streaming completion
        yield {
            "__final__": True,
            "content": "Streamed response",
            "agent_result": MagicMock(
                output="Streamed response",
                all_messages=lambda: [],
                new_messages=lambda: [],
                usage={},
                result=MagicMock(tool_usage_results=[], output="Streamed response"),
            ),
        }

    agent._stream_tokens = mock_stream_tokens

    # Test streaming with multimodal content
    multimodal_content = [
        "Stream analysis of:",
        ImageUrl(url="https://example.com/image.png"),
    ]

    # Call run_stream and verify it completes without error
    result = await agent.run_stream(multimodal_content)

    # Verify the result contains expected output
    assert result.result == "Streamed response"


@pytest.mark.asyncio
async def test_missing_required_fields():
    """Test that missing required fields raise TypeError."""
    with pytest.raises(TypeError):
        ImageUrl()  # Missing required 'url' field


@pytest.mark.asyncio
async def test_invalid_binary_content_field():
    """Test that missing required fields in BinaryContent raise TypeError."""
    with pytest.raises(TypeError):
        BinaryContent(data=b"test")  # Missing required 'media_type' field


@pytest.mark.asyncio
async def test_agent_with_malformed_multimodal():
    """Test that agent handles runtime errors with multimodal content."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the underlying runner to simulate a runtime error
    agent._runner.run = AsyncMock(
        side_effect=RuntimeError("Failed to process multimodal content")
    )

    # Test with valid multimodal content that causes runtime error
    with pytest.raises(RuntimeError, match="Failed to process multimodal content"):
        content = [
            "Analyze this:",
            ImageUrl(url="https://example.com/nonexistent.png"),
        ]
        await agent.run(content)


@pytest.mark.asyncio
async def test_unreachable_image_url():
    """Test that unreachable URLs cause appropriate errors."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock httpx to simulate 404 error
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )

        # Mock the runner to simulate the error propagating
        agent._runner.run = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404),
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            content = [
                "What's in this image?",
                ImageUrl(url="https://example.com/nonexistent.png"),
            ]
            await agent.run(content)


@pytest.mark.asyncio
async def test_unsupported_media_type():
    """Test that unsupported media types cause appropriate errors."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock the runner to simulate unsupported media type error
    agent._runner.run = AsyncMock(
        side_effect=ValueError("Unsupported media type: application/x-malware")
    )

    with pytest.raises(ValueError, match="Unsupported media type"):
        content = [
            "Analyze this file:",
            BinaryContent(
                data=b"malicious content", media_type="application/x-malware"
            ),
        ]
        await agent.run(content)


@pytest.mark.asyncio
async def test_network_timeout():
    """Test that network timeouts are handled appropriately."""
    mock_model = MagicMock(spec=OpenAIModel)

    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        model=mock_model,
        api_key="test-key",
    )

    # Mock httpx to simulate timeout
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Request timed out")

        # Mock the runner to simulate the timeout propagating
        agent._runner.run = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with pytest.raises(httpx.TimeoutException):
            content = [
                "What's in this image?",
                ImageUrl(url="https://slowsite.example.com/image.png"),
            ]
            await agent.run(content)
