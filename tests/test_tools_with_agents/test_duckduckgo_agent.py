from iointel import Agent
from iointel.src.agent_methods.tools.duckduckgo import (
    search_the_web,
)
from iointel.src.utilities.runners import run_agents


def test_duckduckgo():
    agent = Agent(
        name="DuckDuckGo Agent",
        instructions="You are a DuckDuckGo search agent. Use search to respond to the user.",
        tools=[search_the_web],
    )
    result = run_agents(
        "Search the web. How many models were released on the first version of io-intelligence product?",
        agents=[agent],
    ).execute()
    assert "25" in result
