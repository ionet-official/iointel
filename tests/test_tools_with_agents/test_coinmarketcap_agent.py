import asyncio

import marvin

from iointel import Agent
from iointel.src.agent_methods.tools_before_rebase.coinmarketcap import listing_coins, get_coin_info, get_coin_quotes, \
    get_coin_quotes_historical
from iointel.src.agent_methods.tools_before_rebase.utils import what_time_is_it


def get_coinmarketcap_agent():
    return Agent(
        name="Agent",
        instructions="""
            You are the coinmarketcap AI agent.
            You are given functions, which call coinmarketcap API.
            Try to satisfy user's request by figuring out the right endpoint.
            - For requests about particular crypto use 'get_coin_info'
            - When asked calling listing_coins, don't request for more than 10 results.
              - If you need more, split into several requests, aggregate results after each call
            - When asked about yestarday, today, ... all the relative dates. First use the tool to get the current date.
            """,
        tools=[
            listing_coins, get_coin_info, get_coin_quotes,
            get_coin_quotes_historical, what_time_is_it,
        ]
    )

def test_coinmarketcap_btc_year():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "What year was bitcoin established at? Return the date obtained from toolcall result",
            agents=[agent]
        )
    )
    assert result is not None, "Expected a result from the agent run."
    assert '2010' in result or '2009' in result


def test_top_10_currencies_by_capitalization():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "Return names of top 10 cryptocurrencies, sorted by capitalization. "
            "Use the format: currency1,currency2,...,currencyX",
            agents=[agent]
        )
    )
    assert result is not None, "Expected a result from the agent run."
    currencies = result.split(',')
    assert len(currencies) == 10
    assert 'Bitcoin' in currencies
    assert 'Ethereum' in currencies


def test_coinmarketcap_different_crypto_for_same_symbol():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "List some of the cryptocurrency names with a symbol BTC. Use get_coin_info function.",
            agents=[agent]
        )
    )
    assert result is not None, "Expected a result from the agent run."
    assert len(result) > 1
    assert 'Boost Trump Campaign' in result
    assert 'batcat' in result
    assert 'Bullish Trump Coin' in result


def test_coinmarketcap_btc_capitalization():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "What's bitcoin capitalization? Return a single number: capitalization in USD",
            agents=[agent],
            result_type=float
        )
    )
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10**9 # More than 1 billion dollars


def test_coinmarketcap_get_current_price():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "Get current price of bitcoin. Return a single number: price in USD.",
            agents=[agent],
            result_type=float
        )
    )
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10000  # Price should be greater than 10k$


def test_coinmarketcap_historical_price():
    agent = get_coinmarketcap_agent()
    result = asyncio.run(
        marvin.run_async(
            "Get price of bitcoin yesterday at 12:00. Return a single number: price in USD.",
            agents=[agent],
            result_type=float
        )
    )
    assert result is not None, "Expected a result from the agent run."
    assert float(result) > 10000  # Price should be greater than 10k$
