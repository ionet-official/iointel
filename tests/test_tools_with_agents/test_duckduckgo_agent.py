import pytest
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from iointel import Agent
from iointel.src.agent_methods.tools.duckduckgo import (
    search_the_web,
)
from iointel.src.utilities.runners import run_agents


async def test_duckduckgo():
    agent = Agent(
        name="DuckDuckGo Agent",
        instructions="You are a DuckDuckGo search agent. Use search to respond to the user.",
        tools=[search_the_web],
    )
    try:
        result = await run_agents(
            "Search the web. How many models were released on the first version of io-intelligence product?",
            agents=[agent],
        ).execute()
    except DuckDuckGoSearchException as err:
        if "202 Ratelimit" in str(err):
            raise pytest.xfail(reason="DDG rate limited us :(")
        raise
    assert "25" in result
