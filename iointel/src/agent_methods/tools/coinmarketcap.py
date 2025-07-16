import datetime
import os
from typing import Any, Optional, Dict, Literal, Annotated, Union
import httpx
import urllib.parse
from pydantic import Field

from iointel.src.utilities.decorators import register_tool

COINMARKETCAP_API_BASE = "pro-api.coinmarketcap.com"

def get_coinmarketcap_api_key() -> str:
    """Get the CoinMarketCap API key, loading it lazily."""
    key = os.getenv("COINMARKETCAP_API_KEY")
    if not key:
        raise RuntimeError("CoinMarketCap API key is not set - please set COINMARKETCAP_API_KEY environment variable")
    return key


def build_url(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Build the full URL for the CoinMarketCap API request by filtering out None values.
    """
    filtered_params = {k: v for k, v in params.items() if v is not None}
    return f"https://{COINMARKETCAP_API_BASE}/{endpoint}?{urllib.parse.urlencode(filtered_params)}"


def coinmarketcap_request(
    endpoint: str, params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Send a request to the specified CoinMarketCap API endpoint with the given parameters.
    """
    url = build_url(endpoint, params)
    with httpx.Client() as client:
        return make_coinmarketcap_request(client, url)


def make_coinmarketcap_request(client: httpx.Client, url: str) -> dict[str, Any] | None:
    """Make a request to the CoinMarketCap API with proper error handling."""
    try:
        api_key = get_coinmarketcap_api_key()
    except RuntimeError as e:
        print(f"❌ {e}")
        return None
        
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }
    try:
        response = client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@register_tool
def listing_coins(
    start: Annotated[Optional[int], Field(ge=1)] = None,
    limit: Annotated[Optional[int], Field(ge=1, le=5000)] = None,
    price_min: Annotated[Optional[float], Field(ge=0)] = None,
    price_max: Annotated[Optional[float], Field(ge=0)] = None,
    market_cap_min: Annotated[Optional[float], Field(ge=0)] = None,
    market_cap_max: Annotated[Optional[float], Field(ge=0)] = None,
    volume_24h_min: Annotated[Optional[float], Field(ge=0)] = None,
    volume_24h_max: Annotated[Optional[float], Field(ge=0)] = None,
    circulating_supply_min: Annotated[Optional[float], Field(ge=0)] = None,
    circulating_supply_max: Annotated[Optional[float], Field(ge=0)] = None,
    percent_change_24h_min: Annotated[Optional[float], Field(ge=-100)] = None,
    percent_change_24h_max: Annotated[Optional[float], Field(ge=-100)] = None,
    convert: Optional[list[str]] = None,
    convert_id: Optional[list[str]] = None,
    sort: Optional[
        Literal[
            "market_cap",
            "name",
            "symbol",
            "date_added",
            "market_cap_strict",
            "price",
            "circulating_supply",
            "total_supply",
            "max_supply",
            "num_market_pairs",
            "volume_24h",
            "percent_change_1h",
            "percent_change_24h",
            "percent_change_7d",
            "market_cap_by_total_supply_strict",
            "volume_7d",
            "volume_30d",
        ]
    ] = None,
    sort_dir: Optional[Literal["asc", "desc"]] = None,
    cryptocurrency_type: Optional[Literal["all", "coins", "tokens"]] = None,
    tag: Optional[Literal["all", "defi", "filesharing"]] = None,
    aux: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a paginated list of active cryptocurrencies with the latest market data from CoinMarketCap.

    Parameters:
        start: Offset the start of the paginated list.
        limit: Specify the number of results to return.
        price_min: Filter results by minimum USD price.
        price_max: Filter results by maximum USD price.
        market_cap_min: Filter results by minimum market cap.
        market_cap_max: Filter results by maximum market cap.
        volume_24h_min: Filter results by minimum 24-hour USD volume.
        volume_24h_max: Filter results by maximum 24-hour USD volume.
        circulating_supply_min: Filter results by minimum circulating supply.
        circulating_supply_max: Filter results by maximum circulating supply.
        percent_change_24h_min: Filter results by minimum 24-hour percent change.
        percent_change_24h_max: Filter results by maximum 24-hour percent change.
        convert: Calculate market quotes in multiple currencies using a list of symbols.
        convert_id: Calculate market quotes by CoinMarketCap ID.
        sort: Field to sort the list of cryptocurrencies.
        sort_dir: Direction to sort the results.
        cryptocurrency_type: Filter by cryptocurrency type.
        tag: Filter by cryptocurrency tag.
        aux: Specify supplemental data fields to return.
             Valid values include ["num_market_pairs", "cmc_rank", "date_added", "tags", "platform", "max_supply", "circulating_supply", "total_supply", "is_active", "is_fiat"].

    Returns:
        A dictionary containing the cryptocurrency listing data if successful, or None otherwise.
    """
    params = {
        "start": start,
        "limit": limit,
        "price_min": price_min,
        "price_max": price_max,
        "market_cap_min": market_cap_min,
        "market_cap_max": market_cap_max,
        "volume_24h_min": volume_24h_min,
        "volume_24h_max": volume_24h_max,
        "circulating_supply_min": circulating_supply_min,
        "circulating_supply_max": circulating_supply_max,
        "percent_change_24h_min": percent_change_24h_min,
        "percent_change_24h_max": percent_change_24h_max,
        "convert": ",".join(convert) if convert else None,
        "convert_id": ",".join(convert_id) if convert_id else None,
        "sort": sort,
        "sort_dir": sort_dir,
        "cryptocurrency_type": cryptocurrency_type,
        "tag": tag,
        "aux": ",".join(aux) if aux else None,
    }

    return coinmarketcap_request("v1/cryptocurrency/listings/latest", params)


def _parse_triplet(
    id: Optional[list[str]] = None,
    slug: Optional[list[str]] = None,
    symbol: Optional[list[str]] = None,
) -> dict:
    if id:
        slug = symbol = None
    elif slug:
        symbol = None
    return {
        "id": ",".join(id) if id else None,
        "slug": ",".join(slug) if slug else None,
        "symbol": ",".join(symbol) if symbol else None,
    }


@register_tool
def get_coin_info(
    id: Optional[list[str]] = None,
    slug: Optional[list[str]] = None,
    symbol: Optional[list[str]] = None,
    address: Optional[str] = None,
    skip_invalid: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve coin information including details such as logo, description, official website URL,
    social links, and links to technical documentation.

    Parameters:
        id: A list of cryptocurrency CoinMarketCap IDs. Example: ["1" , "2"].
        slug: Alternatively pass a list of cryptocurrency slugs. Example: ["bitcoin", "ethereum"].
        symbol: Alternatively pass a list of cryptocurrency symbols. Example: ["BTC", "ETH"].
        address: A contract address for the cryptocurrency. Example: "0xc40af1e4fecfa05ce6bab79dcd8b373d2e436c4e".
        skip_invalid: When True, invalid cryptocurrency lookups will be skipped instead of raising an error.

    Returns:
        A dictionary containing the coin information if the request is successful, or None otherwise.
    """
    params = _parse_triplet(id, slug, symbol) | {
        "address": address,
        "skip_invalid": skip_invalid,
    }

    return coinmarketcap_request("v2/cryptocurrency/info", params)


@register_tool
def get_coin_quotes(
    id: Optional[list[str]] = None,
    slug: Optional[list[str]] = None,
    symbol: Optional[list[str]] = None,
    convert: Optional[list[str]] = None,
    convert_id: Optional[list[str]] = None,
    aux: Optional[list[str]] = None,
    skip_invalid: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest market quote for one or more cryptocurrencies.
    Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.

    Parameters:
        id: A list of cryptocurrency CoinMarketCap IDs. Example: ["1" , "2"].
        slug: Alternatively pass a list of cryptocurrency slugs. Example: ["bitcoin", "ethereum"].
        symbol: Alternatively pass a list of cryptocurrency symbols. Example: ["BTC", "ETH"].
        convert: Optionally calculate market quotes in up to 120 currencies at once by passing a list of cryptocurrency or fiat currency symbols.
        convert_id: Optionally calculate market quotes by CoinMarketCap ID instead of symbol.
        aux: Optionally specify a list of supplemental data fields to return.
             Valid values include ["num_market_pairs", "cmc_rank", "date_added", "tags", "platform", "max_supply", "circulating_supply", "total_supply", "is_active", "is_fiat"].
        skip_invalid: Pass true to relax request validation rules.

    Returns:
        A dictionary containing the latest market quote data if the request is successful, or None otherwise.
    """
    params = _parse_triplet(id, slug, symbol) | {
        "convert": ",".join(convert) if convert else None,
        "convert_id": ",".join(convert_id) if convert_id else None,
        "aux": ",".join(aux) if aux else None,
        "skip_invalid": skip_invalid,
    }
    return coinmarketcap_request("v2/cryptocurrency/quotes/latest", params)


@register_tool
def get_coin_quotes_historical(
    id: Optional[list[str]] = None,
    slug: Optional[list[str]] = None,
    symbol: Optional[list[str]] = None,
    convert: Optional[list[str]] = None,
    convert_id: Optional[list[str]] = None,
    aux: Optional[list[str]] = None,
    skip_invalid: bool = False,
    time_start: Optional[Union[datetime.datetime, str]] = None,
    time_end: Optional[Union[datetime.datetime, str]] = None,
    count: int = 10,
    interval: str = "5m",
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest market quote for one or more cryptocurrencies.
    Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.

    To get historical price at a particular point of time, provide time_end=<point-of-time> and count=1

    Parameters:
        id: A list of cryptocurrency CoinMarketCap IDs. Example: ["1" , "2"].
        slug: Alternatively pass a list of cryptocurrency slugs. Example: ["bitcoin", "ethereum"].
        symbol: Alternatively pass a list of cryptocurrency symbols. Example: ["BTC", "ETH"].
        convert: Optionally calculate market quotes in up to 120 currencies at once by passing a list of cryptocurrency or fiat currency symbols.
        convert_id: Optionally calculate market quotes by CoinMarketCap ID instead of symbol.
        aux: Optionally specify a list of supplemental data fields to return.
             Valid values include ["num_market_pairs", "cmc_rank", "date_added", "tags", "platform", "max_supply", "circulating_supply", "total_supply", "is_active", "is_fiat"].
        skip_invalid: Pass true to relax request validation rules.
        time_start: timestamp to start returning quotes for.
                    Optional, if not passed, we'll return quotes calculated in reverse from "time_end".
        time_end: timestamp to stop returning quotes for (inclusive).
                  Optional, if not passed, we'll default to the current time.
                  If no "time_start" is passed, we return quotes in reverse order starting from this time.
        count: The number of interval periods to return results for.
               Optional, required if both "time_start" and "time_end" aren't supplied.
               The default is 10 items. The current query limit is 10000.
        interval: Interval of time to return data points for. See details in endpoint description.

    Returns:
        A dictionary containing the latest market quote data if the request is successful, or None otherwise.
    """
    # Handle both string and datetime inputs for time parameters
    if time_start:
        if isinstance(time_start, str):
            time_start = time_start
        else:
            time_start = time_start.replace(microsecond=0).isoformat()
    else:
        time_start = None
        
    if time_end:
        if isinstance(time_end, str):
            time_end = time_end
        else:
            time_end = time_end.replace(microsecond=0).isoformat()
    else:
        time_end = None
    params = _parse_triplet(id, slug, symbol) | {
        "convert": ",".join(convert) if convert else None,
        "convert_id": ",".join(convert_id) if convert_id else None,
        "aux": ",".join(aux) if aux else None,
        "skip_invalid": skip_invalid,
        "time_start": time_start,
        "time_end": time_end,
        "count": count,
        "interval": interval,
    }
    return coinmarketcap_request("v2/cryptocurrency/quotes/historical", params)
