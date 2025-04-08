import datetime
import json

from iointel.src.agent_methods.tools.coinmarketcap import (
    get_coin_quotes_historical,
    get_coin_quotes,
    get_coin_info,
    listing_coins,
)


async def test_listing_coins():
    assert await listing_coins()


async def test_get_coin_info():
    assert await get_coin_info(symbol=["BTC"])


async def test_get_coin_price():
    assert await get_coin_quotes(symbol=["BTC"])


async def test_get_coin_historical_price():
    assert json.dumps(
        await get_coin_quotes_historical(
            symbol=["BTC"],
            time_end=datetime.datetime(
                year=2025, month=3, day=17, hour=12, minute=0, second=0
            ),
            count=1,
        ),
        indent=4,
        sort_keys=True,
    )
