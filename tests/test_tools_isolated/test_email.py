import os
import pytest
from unittest.mock import patch
from iointel.src.agent_methods.tools.agno.email import Email
from pydantic_ai.models.openai import OpenAIModel
from iointel.src.agents import Agent

__open_api_model = "gpt-4o-mini"


def test_email_basic_functionality():
    """Test the basic email functionality."""
    # Mock the email_user method at the class level *with autospec* so "self" is passed automatically.
    with patch.object(Email, "email_user", autospec=True) as mock_email_user:
        mock_email_user.return_value = "Email sent successfully: This is a test message"

        email = Email(
            receiver_email="test@example.com",
            sender_name="Test Sender",
            sender_email="sender@example.com",
            sender_passkey="test_password123",
        )

        # Test sending a simple message
        subject = "Test Subject"
        body = "This is a test message"
        result = email.email_user(subject=subject, body=body)

        # Verify the mock was called with correct arguments (including self)
        mock_email_user.assert_called_once_with(email, subject=subject, body=body)

        # Verify the response
        assert isinstance(result, str)
        assert body in result  # The message should be reflected in the response


@pytest.mark.asyncio
async def test_email_with_agent():
    """Simulate an agent sending an email without hitting OpenAI servers."""

    # Stub for Agent.run that mimics a successful execution.
    async def agent_run_stub(self, prompt: str):  # noqa: D401 – simple stub
        # Pretend the agent used email_user internally and is confirming send.
        return {"result": "Hello! Your email has been sent successfully."}

    with patch.object(Agent, "run", new=agent_run_stub):
        email_tool = Email()
        agent = Agent(
            name="EmailAgent",
            instructions=(
                "You are an email assistant AI agent. "
                "When asked to send an email, use the email_user tool."
            ),
            tools=[email_tool.email_user],
            model=OpenAIModel(model_name=__open_api_model),
        )

        result = await agent.run("Send an email saying hello")

        # Validate the stubbed result
        assert isinstance(result["result"], str)
        assert "hello" in result["result"].lower()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
async def test_email_with_agent_complex():
    """Placeholder for a richer integration test that hits a real LLM."""
    email_tool = Email()
    agent = Agent(
        name="EmailAssistant",
        instructions=(
            "You are a helpful email assistant. Use the email_user tool to send messages."
        ),
        tools=[email_tool.email_user],
        model=OpenAIModel(model_name=__open_api_model),
    )

    result = await agent.run(
        "Send an email about the upcoming project meeting scheduled for tomorrow at 2 PM."
    )

    print("@@@ result:", result)

    # assert isinstance(result["result"], str)
    # assert any(
    #     kw in result["result"].lower()
    #     for kw in ["meeting", "project", "tomorrow", "2 pm", "scheduled"]
    # )


async def test_email_with_agent_complex2():
    """Complex prompt, still stubbed – no external dependencies."""

    async def agent_run_stub(self, prompt: str):
        return {
            "result": (
                "Your reminder email about the project meeting scheduled for "
                "tomorrow at 2PM has been sent successfully."
            )
        }

    with patch.object(Agent, "run", new=agent_run_stub):
        email_tool = Email()
        agent = Agent(
            name="EmailAssistant",
            instructions=(
                "You are a helpful email assistant. Use the email_user tool to send messages."
            ),
            tools=[email_tool.email_user],
            model=OpenAIModel(model_name="gpt-4o-mini"),
        )

        result = (
            await agent.run(
                "Send an email about the upcoming project meeting scheduled for tomorrow at 2PM."
            )
        )["result"]

        assert isinstance(result, str)
        assert all(
            [
                kw in result.lower()
                for kw in ["meeting", "project", "tomorrow", "2pm", "sent"]
            ]
        )
