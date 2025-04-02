import asyncio

from iointel import Agent
from iointel.src.agent_methods.tools_before_rebase.searxng import SearxngClient
from iointel.src.workflow import run_agents_async


def test_searxng():
    client = SearxngClient(base_url='http://localhost:8080')

    agent = Agent(
        name="Agent",
        instructions="You are a Searxng search agent. Use search to respond to the user.",
        tools=[client.search],
    )

    result = asyncio.run(run_agents_async(
        "Search the web. How many models were released on the first version of io-intelligence product?",
        agents=[agent],
    ))

    assert '25' in result