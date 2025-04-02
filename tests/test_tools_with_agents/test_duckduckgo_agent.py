import asyncio

from iointel import Agent
from iointel.src.agent_methods.tools_before_rebase.duckduckgo import search_the_web_async
from iointel.src.utilities.runners import run_agents_async


def test_duckduckgo():
    agent = Agent(
        name="DuckDuckGo Agent",
        instructions="You are a DuckDuckGo search agent. Use search to respond to the user.",
        tools=[search_the_web_async],
    )
    result = asyncio.run(run_agents_async(
        "Search the web. How many models were released on the first version of io-intelligence product?",
        agents=[agent]
    ))
    assert '25' in result
