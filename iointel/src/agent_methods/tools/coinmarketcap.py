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
    
    Parameters:
        endpoint: API endpoint path (e.g., "v1/cryptocurrency/listings/latest")
        params: Dictionary of query parameters to include in the URL
        
    Returns:
        Complete URL with encoded query parameters
    """
    filtered_params = {k: v for k, v in params.items() if v is not None}
    return f"https://{COINMARKETCAP_API_BASE}/{endpoint}?{urllib.parse.urlencode(filtered_params)}"


def coinmarketcap_request(
    endpoint: str, params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Send a request to the specified CoinMarketCap API endpoint with the given parameters.
    
    Parameters:
        endpoint: API endpoint path (e.g., "v1/cryptocurrency/listings/latest")
        params: Dictionary of query parameters for the API request
        
    Returns:
        JSON response from the API as a dictionary, or None if request fails
    """
    url = build_url(endpoint, params)
    with httpx.Client() as client:
        return make_coinmarketcap_request(client, url)


def make_coinmarketcap_request(client: httpx.Client, url: str) -> dict[str, Any] | None:
    """
    Make a request to the CoinMarketCap API with proper error handling.
    
    Parameters:
        client: HTTP client to use for the request
        url: Full URL to make the request to
        
    Returns:
        JSON response as a dictionary, or None if request fails
        
    Note:
        Requires COINMARKETCAP_API_KEY environment variable to be set
    """
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
    except Exception as e:
        print(f"❌ CoinMarketCap API request failed: {e}")
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
    id: Optional[Union[str, list[str]]] = None,
    slug: Optional[Union[str, list[str]]] = None,
    symbol: Optional[Union[str, list[str]]] = None,
) -> dict:
    """
    Parse cryptocurrency identifiers, converting strings to lists if needed.
    
    Parameters:
        id: Cryptocurrency CoinMarketCap ID(s) - can be a single ID or list
        slug: Cryptocurrency slug(s) - can be a single slug or list  
        symbol: Cryptocurrency symbol(s) - can be a single symbol or list
        
    Returns:
        Dictionary with parsed and formatted identifiers
        
    Note:
        Only one type of identifier should be provided. If multiple are given,
        priority order is: id > slug > symbol
    """
    # Convert strings to lists for consistent handling
    if isinstance(id, str):
        id = [id]
    if isinstance(slug, str):
        slug = [slug]
    if isinstance(symbol, str):
        symbol = [symbol]
        
    # Priority: id > slug > symbol
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
    id: Optional[Union[str, list[str]]] = None,
    slug: Optional[Union[str, list[str]]] = None,
    symbol: Optional[Union[str, list[str]]] = None,
    address: Optional[str] = None,
    skip_invalid: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve coin information including details such as logo, description, official website URL,
    social links, and links to technical documentation.

    Parameters:
        id: Cryptocurrency CoinMarketCap ID(s). Can be a single ID ("1") or list (["1", "2"])
        slug: Cryptocurrency slug(s). Can be a single slug ("bitcoin") or list (["bitcoin", "ethereum"])
        symbol: Cryptocurrency symbol(s). Can be a single symbol ("BTC") or list (["BTC", "ETH"])
        address: A contract address for the cryptocurrency. Example: "0xc40af1e4fecfa05ce6bab79dcd8b373d2e436c4e"
        skip_invalid: When True, invalid cryptocurrency lookups will be skipped instead of raising an error

    Returns:
        A dictionary containing the coin information if the request is successful, or None otherwise
        
    Examples:
        >>> get_coin_info(symbol="BTC")  # Single symbol
        >>> get_coin_info(symbol=["BTC", "ETH"])  # Multiple symbols
        >>> get_coin_info(id="1")  # Single ID
        >>> get_coin_info(slug=["bitcoin", "ethereum"])  # Multiple slugs
    """
    params = _parse_triplet(id, slug, symbol) | {
        "address": address,
        "skip_invalid": skip_invalid,
    }

    return coinmarketcap_request("v2/cryptocurrency/info", params)


@register_tool
def get_coin_quotes(
    id: Optional[Union[str, list[str]]] = None,
    slug: Optional[Union[str, list[str]]] = None,
    symbol: Optional[Union[str, list[str]]] = None,
    convert: Optional[Union[str, list[str]]] = None,
    convert_id: Optional[Union[str, list[str]]] = None,
    aux: Optional[Union[str, list[str]]] = None,
    skip_invalid: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest market quote for one or more cryptocurrencies.
    Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.

    Parameters:
        id: Cryptocurrency CoinMarketCap ID(s). Can be a single ID ("1") or list (["1", "2"])
        slug: Cryptocurrency slug(s). Can be a single slug ("bitcoin") or list (["bitcoin", "ethereum"])
        symbol: Cryptocurrency symbol(s). Can be a single symbol ("BTC") or list (["BTC", "ETH"])
        convert: Currency/currencies to convert prices to. Can be single ("USD") or list (["USD", "EUR"])
        convert_id: CoinMarketCap ID(s) to convert prices to instead of symbols
        aux: Supplemental data fields to return. Valid values include:
             ["num_market_pairs", "cmc_rank", "date_added", "tags", "platform", 
              "max_supply", "circulating_supply", "total_supply", "is_active", "is_fiat"]
        skip_invalid: Pass true to relax request validation rules

    Returns:
        A dictionary containing the latest market quote data if successful, or None otherwise
        
    Examples:
        >>> get_coin_quotes(symbol="BTC")  # Get BTC price
        >>> get_coin_quotes(symbol=["BTC", "ETH"])  # Get multiple prices
        >>> get_coin_quotes(symbol="BTC", convert="EUR")  # Get BTC in EUR
        >>> get_coin_quotes(symbol=["BTC", "ETH"], convert=["USD", "EUR"])  # Multiple conversions
    """
    # Convert single strings to lists for consistent handling
    if isinstance(convert, str):
        convert = [convert]
    if isinstance(convert_id, str):
        convert_id = [convert_id]
    if isinstance(aux, str):
        aux = [aux]
        
    params = _parse_triplet(id, slug, symbol) | {
        "convert": ",".join(convert) if convert else None,
        "convert_id": ",".join(convert_id) if convert_id else None,
        "aux": ",".join(aux) if aux else None,
        "skip_invalid": skip_invalid,
    }
    return coinmarketcap_request("v2/cryptocurrency/quotes/latest", params)


@register_tool
def get_coin_quotes_historical(
    id: Optional[Union[str, list[str]]] = None,
    slug: Optional[Union[str, list[str]]] = None,
    symbol: Optional[Union[str, list[str]]] = None,
    convert: Optional[Union[str, list[str]]] = None,
    convert_id: Optional[Union[str, list[str]]] = None,
    aux: Optional[Union[str, list[str]]] = None,
    skip_invalid: bool = False,
    time_start: Optional[Union[datetime.datetime, str]] = None,
    time_end: Optional[Union[datetime.datetime, str]] = None,
    count: int = 10,
    interval: str = "5m",
) -> Optional[Dict[str, Any]]:
    """
    Retrieve historical market quotes for one or more cryptocurrencies.
    Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions.

    To get historical price at a particular point of time, provide time_end=<point-of-time> and count=1

    Parameters:
        id: Cryptocurrency CoinMarketCap ID(s). Can be a single ID ("1") or list (["1", "2"])
        slug: Cryptocurrency slug(s). Can be a single slug ("bitcoin") or list (["bitcoin", "ethereum"])
        symbol: Cryptocurrency symbol(s). Can be a single symbol ("BTC") or list (["BTC", "ETH"])
        convert: Currency/currencies to convert prices to. Can be single ("USD") or list (["USD", "EUR"])
        convert_id: CoinMarketCap ID(s) to convert prices to instead of symbols
        aux: Supplemental data fields to return. Valid values include:
             ["num_market_pairs", "cmc_rank", "date_added", "tags", "platform",
              "max_supply", "circulating_supply", "total_supply", "is_active", "is_fiat"]
        skip_invalid: Pass true to relax request validation rules
        time_start: Starting timestamp for quotes (datetime or ISO string).
                    If not provided, returns quotes in reverse from time_end
        time_end: Ending timestamp for quotes (datetime or ISO string).
                  Defaults to current time if not provided
        count: Number of interval periods to return (default: 10, max: 10000).
               Required if both time_start and time_end aren't supplied
        interval: Time interval between data points. Options include:
                  "5m", "10m", "15m", "30m", "45m", "1h", "2h", "3h", 
                  "6h", "12h", "24h", "1d", "2d", "3d", "7d", "14d", "15d", 
                  "30d", "60d", "90d", "365d"

    Returns:
        A dictionary containing historical market quote data if successful, or None otherwise
        
    Examples:
        >>> # Get last 10 5-minute intervals for BTC
        >>> get_coin_quotes_historical(symbol="BTC")
        
        >>> # Get BTC price at specific time
        >>> get_coin_quotes_historical(symbol="BTC", time_end="2024-01-01T00:00:00", count=1)
        
        >>> # Get daily prices for last 30 days
        >>> get_coin_quotes_historical(symbol=["BTC", "ETH"], interval="1d", count=30)
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
        
    # Convert single strings to lists for consistent handling
    if isinstance(convert, str):
        convert = [convert]
    if isinstance(convert_id, str):
        convert_id = [convert_id]
    if isinstance(aux, str):
        aux = [aux]
        
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
