import pytest

from iointel import Agent
from iointel.src.agent_methods.tools.coinmarketcap import (
    listing_coins,
    get_coin_info,
    get_coin_quotes,
    get_coin_quotes_historical,
)
from iointel.src.agent_methods.tools.utils import what_time_is_it
from iointel.src.utilities.runners import run_agents


@pytest.fixture
def coinmarketcap_agent():
    return Agent(
        name="Agent",
        instructions="""
            You are the coinmarketcap AI agent.
            Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.

            You are given functions which call coinmarketcap API.
            Try to satisfy user's request by figuring out the right endpoint.
            - For requests about particular crypto call 'get_coin_info' function.
            - When asked calling listing_coins, don't request for more than 10 results.
              - If you need more, split into several requests, aggregate results after each call
            - When asked about yesterday, today, ... all the relative dates, first use the tool to get the current date.
            """,
        tools=[
            listing_coins,
            get_coin_info,
            get_coin_quotes,
            get_coin_quotes_historical,
            what_time_is_it,
        ],
    )


async def test_coinmarketcap_btc_year(coinmarketcap_agent):
    result = await run_agents(
        "What year was bitcoin established at? Return the date obtained from toolcall result",
        agents=[coinmarketcap_agent],
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    assert "2010" in result or "2009" in result


async def test_top_10_currencies_by_capitalization(coinmarketcap_agent):
    result = await run_agents(
        "Return names of top 10 cryptocurrencies, sorted by capitalization. "
        "Use the format: currency1,currency2,...,currencyX",
        agents=[coinmarketcap_agent],
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    currencies = result.split(",")
    assert len(currencies) == 10
    assert "Bitcoin" in currencies
    assert "Ethereum" in currencies


async def test_coinmarketcap_different_crypto_for_same_symbol(coinmarketcap_agent):
    result = await run_agents(
        "List some of the cryptocurrency names with a symbol BTC. Use get_coin_info function.",
        agents=[coinmarketcap_agent],
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    assert len(result) > 1
    assert "Boost Trump Campaign" in result
    assert "batcat" in result
    assert "Bullish Trump Coin" in result


async def test_coinmarketcap_btc_capitalization(coinmarketcap_agent):
    result = await run_agents(
        "What's bitcoin capitalization? Return a single number: capitalization in USD",
        agents=[coinmarketcap_agent],
        output_type=float,
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10**9  # More than 1 billion dollars


async def test_coinmarketcap_get_current_price(coinmarketcap_agent):
    result = await run_agents(
        "Get current price of bitcoin. Return a single number: price in USD.",
        agents=[coinmarketcap_agent],
        output_type=float,
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10000  # Price should be greater than 10k$


# Looks like it fails to pass an array into function params
# Maybe we can wait for LLama 4 to fix things
async def test_coinmarketcap_historical_price(coinmarketcap_agent):
    result = await run_agents(
        "Get price of bitcoin yesterday at 12:00. Return a single number: price in USD.",
        agents=[coinmarketcap_agent],
        output_type=float,
    ).execute()
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10000  # Price should be greater than 10k$
