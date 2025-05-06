import os

import pytest

from iointel import Agent
from iointel.src.agent_methods.tools.searxng import SearxngClient
from iointel.src.utilities.runners import run_agents


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Coudn't run searxng in github CI"
)
async def test_searxng():
    client = SearxngClient(base_url="http://localhost:8080")

    agent = Agent(
        name="Agent",
        instructions="You are a Searxng search agent. Use search to respond to the user.",
        tools=[client.search],
    )

    result = (await run_agents(
        "Search the web. How many models were released on the first version of io-intelligence product?",
        agents=[agent],
    )).execute()

    assert "25" in result
