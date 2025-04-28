import datetime

import pytest

from iointel.src.agent_methods.tools.coinmarketcap import (
    get_coin_quotes_historical,
    get_coin_quotes,
    get_coin_info,
    listing_coins,
)


def test_listing_coins():
    assert listing_coins()


def test_get_coin_info():
    assert get_coin_info(symbol=["BTC"])


def test_get_coin_price():
    assert get_coin_quotes(symbol=["BTC"])


@pytest.mark.skip(reason="Waiting to get a paid coinmarketcap API key")
def test_get_coin_historical_price():
    assert get_coin_quotes_historical(
        symbol=["BTC"],
        time_end=datetime.datetime.now() - datetime.timedelta(days=1),
        count=1,
    )
