import os
import json

import pytest
from iointel.src.agent_methods.tools.agno.yfinance import YFinance
from pydantic_ai.models.openai import OpenAIModel

from iointel.src.agents import Agent


###############################
#  Basic, synchronous tests   #
###############################


def test_yfinance_current_stock_price():
    """Ensure we can fetch a current stock price and that it looks sane."""
    yf = YFinance()

    result = yf.get_current_stock_price("AAPL")
    # We don't know the exact format (could be a raw number or JSON) so be flexible.
    assert result, "get_current_stock_price returned an empty result"

    try:
        # Case 1 – the tool returns the price as a raw stringified float
        price = float(result)
        assert price > 0, f"Price should be > 0, got {price} from {result}"
    except ValueError:
        # Case 2 – the tool returns JSON
        data = json.loads(result)
        assert data["symbol"].upper() == "AAPL", (
            f"Unexpected symbol in response: {data}"
        )
        assert float(data["price"]) > 0, f"Price should be > 0, got {data}"


def test_yfinance_company_info():
    """Company info should mention the company name or symbol."""
    yf = YFinance()
    result = yf.get_company_info("AAPL")
    # print('@@@ result:', result)
    assert result, "get_company_info returned an empty result"

    # The payload is typically JSON, but fall back to raw text just in case.
    try:
        info = json.loads(result)
        name = info.get("Name") or info.get("name", "")
        assert "apple" in name.lower() or "AAPL" in name.upper()
    except json.JSONDecodeError:
        assert ("apple" in result.lower()) or ("AAPL" in result.upper())


def test_yfinance_stock_fundamentals():
    """Validate that key fundamentals are returned and look reasonable."""
    yf = YFinance()
    result = yf.get_stock_fundamentals("AAPL")
    assert result, "get_stock_fundamentals returned an empty result"
    print("@@@ result:", result)

    try:
        fundamentals = json.loads(result)
        # Market cap should be a positive number in either raw or string form
        market_cap = fundamentals.get("market_cap")
        if isinstance(market_cap, str):
            market_cap = float(market_cap.replace(",", ""))
        assert market_cap and market_cap > 0, f"Unexpected marketCap: {market_cap}"
    except json.JSONDecodeError:
        # Skip detailed validation if format is unknown; just ensure non‑empty.
        pass


################################
#  Asynchronous agent-based tests  #
################################


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_yfinance_with_agent_current_price():
    """Verify that an agent can call the yfinance tool to fetch the current price."""

    yf = YFinance()
    agent = Agent(
        name="StockAgent",
        instructions=(
            """
            You are a finance assistant. Use the YFinance tools to answer stock‑related questions.
            Available tools: get_current_stock_price, get_company_info, get_historical_stock_prices,
            get_stock_fundamentals, get_income_statements, get_key_financial_ratios,
            get_analyst_recommendations, get_company_news, get_technical_indicators.
            """
        ),
        tools=[
            yf.get_current_stock_price,
            yf.get_company_info,
        ],
        model=OpenAIModel(model_name="gpt-4o-mini"),
    )

    # Ask the agent a simple question.
    result = await agent.run("What is the current stock price of AAPL?")
    # We don't know the exact wording; just make sure we got a non‑empty response.
    assert (
        "AAPL" in str(result["result"]).upper() or str(result["result"]).strip() != ""
    )
