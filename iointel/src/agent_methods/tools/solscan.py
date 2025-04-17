import json
import os
from typing import Any, Dict, List, Optional, Literal, Tuple, Annotated


import httpx

# Replace these with your actual API key and wallet address
SOLSCAN_API_KEY = os.getenv("SOLSCAN_API_KEY")

# Base URL for Solscan Pro API V2
SOLSCAN_API_URL = "https://pro-api.solscan.io"


def fetch_solscan(
    path, full_url: Optional[str] = None, params: Optional[dict] = None
) -> dict | str:
    if not SOLSCAN_API_KEY:
        raise RuntimeError("Solscan API key is not set")
    url = full_url if full_url else f"{SOLSCAN_API_URL}{path}"
    headers = {"token": SOLSCAN_API_KEY}
    with httpx.Client() as client:
        response = client.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return response.text


# Define common type aliases for readability and reuse
TransferActivityType = Literal[
    "ACTIVITY_SPL_TRANSFER",
    "ACTIVITY_SPL_BURN",
    "ACTIVITY_SPL_MINT",
    "ACTIVITY_SPL_CREATE_ACCOUNT",
]
DefiActivityType = Literal[
    "ACTIVITY_TOKEN_SWAP",
    "ACTIVITY_AGG_TOKEN_SWAP",
    "ACTIVITY_TOKEN_ADD_LIQ",
    "ACTIVITY_TOKEN_REMOVE_LIQ",
    "ACTIVITY_SPL_TOKEN_STAKE",
    "ACTIVITY_SPL_TOKEN_UNSTAKE",
    "ACTIVITY_SPL_TOKEN_WITHDRAW_STAKE",
    "ACTIVITY_SPL_INIT_MINT",
]
NftActivityType = Literal[
    "ACTIVITY_NFT_SOLD",
    "ACTIVITY_NFT_LISTING",
    "ACTIVITY_NFT_BIDDING",
    "ACTIVITY_NFT_CANCEL_BID",
    "ACTIVITY_NFT_CANCEL_LIST",
    "ACTIVITY_NFT_REJECT_BID",
    "ACTIVITY_NFT_UPDATE_PRICE",
    "ACTIVITY_NFT_LIST_AUCTION",
]

PageSizeSmall = Literal[10, 20, 30, 40]
PageSizeMedium = Literal[10, 20, 30, 40, 60, 100]
PageSizeNft = Literal[12, 24, 36]
PageSizeCollection = Literal[10, 18, 20, 30, 40]

SortOrder = Literal["asc", "desc"]
TokenAccountType = Literal["token", "nft"]
VoteFilter = Literal["exceptVote", "all"]
AddressList5 = Annotated[List[str], "max_length 5"]

DateYYYYMMDD = Annotated[str, "format YYYYMMDD"]


# Account APIs
def fetch_account_detail(address: str) -> dict | str:
    """Get the details of an account."""
    params = {"address": address}
    return fetch_solscan("/v2.0/account/detail", params=params)


def fetch_account_transfer(
    address: str,
    activity_type: Optional[List[TransferActivityType]] = None,
    token_account: Optional[str] = None,
    from_address: Optional[str] = None,
    to_address: Optional[str] = None,
    token: Optional[str] = None,
    amount: Optional[Tuple[float, float]] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    exclude_amount_zero: Optional[bool] = None,
) -> dict | str:
    """Get transfer data of an account (with optional filters)."""
    params: Dict[str, Any] = {"address": address}
    if activity_type is not None:
        params["activity_type"] = activity_type
    if token_account is not None:
        params["token_account"] = token_account
    if from_address is not None:
        params["from"] = from_address
    if to_address is not None:
        params["to"] = to_address
    if token is not None:
        params["token"] = token
    if amount is not None:
        # Expect tuple (min, max) amount range
        params["amount"] = list(amount)
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if exclude_amount_zero is not None:
        params["exclude_amount_zero"] = exclude_amount_zero
    return fetch_solscan("/v2.0/account/transfer", params=params)


def fetch_account_defi_activities(
    address: str,
    activity_type: Optional[List[DefiActivityType]] = None,
    from_address: Optional[str] = None,
    platform: Optional[AddressList5] = None,
    source: Optional[AddressList5] = None,
    token: Optional[str] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
) -> dict | str:
    """Get DeFi activities involving an account."""
    params: Dict[str, Any] = {"address": address}
    if activity_type is not None:
        params["activity_type"] = activity_type
    if from_address is not None:
        params["from"] = from_address
    if platform is not None:
        params["platform"] = platform
    if source is not None:
        params["source"] = source
    if token is not None:
        params["token"] = token
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/account/defi/activities", params=params)


def fetch_account_balance_change_activities(
    address: str,
    token_account: Optional[str] = None,
    token: Optional[str] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
    page: Optional[int] = None,
    remove_spam: Optional[bool] = None,
    amount: Optional[Tuple[float, float]] = None,
    flow: Optional[Literal["in", "out"]] = None,
) -> dict | str:
    """Get balance change activities (token inflows/outflows) for an account."""
    params: Dict[str, Any] = {"address": address}
    if token_account is not None:
        params["token_account"] = token_account
    if token is not None:
        params["token"] = token
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if page_size is not None:
        params["page_size"] = page_size
    if page is not None:
        params["page"] = page
    if remove_spam is not None:
        params["remove_spam"] = "true" if remove_spam else "false"
    if amount is not None:
        params["amount"] = list(amount)
    if flow is not None:
        params["flow"] = flow
    return fetch_solscan("/v2.0/account/balance_change", params=params)


def fetch_account_transactions(
    address: str,
    before: Optional[str] = None,
    limit: Optional[Literal[10, 20, 30, 40]] = None,
) -> dict | str:
    """Get list of transactions for an account (with pagination)."""
    params: Dict[str, Any] = {"address": address}
    if before is not None:
        params["before"] = before
    if limit is not None:
        params["limit"] = limit
    return fetch_solscan("/v2.0/account/transactions", params=params)


def fetch_account_portfolio(address: str) -> dict | str:
    """Get the portfolio (token balances and values) for a given address."""
    params = {"address": address}
    return fetch_solscan("/v2.0/account/portfolio", params=params)


def fetch_account_token_accounts(
    address: str,
    account_type: TokenAccountType,
    page: Optional[int] = None,
    page_size: Optional[PageSizeSmall] = None,
    hide_zero: Optional[bool] = None,
) -> dict | str:
    """Get token accounts of an address (either SPL tokens or NFTs)."""
    params: Dict[str, Any] = {"address": address, "type": account_type}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if hide_zero is not None:
        params["hide_zero"] = hide_zero
    return fetch_solscan("/v2.0/account/token-accounts", params=params)


def fetch_account_stake(
    address: str, page: Optional[int] = None, page_size: Optional[PageSizeSmall] = None
) -> dict | str:
    """Get the list of stake accounts for a given wallet address."""
    params: Dict[str, Any] = {"address": address}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/account/stake", params=params)


def fetch_stake_rewards_export(
    address: str, time_from: Optional[int] = None, time_to: Optional[int] = None
) -> dict | str:
    """Export staking reward history for an account (up to 5000 records)."""
    params: Dict[str, Any] = {"address": address}
    if time_from is not None:
        params["time_from"] = time_from
    if time_to is not None:
        params["time_to"] = time_to
    return fetch_solscan("/v2.0/account/reward/export", params=params)


def fetch_account_transfer_export(
    address: str,
    activity_type: Optional[List[TransferActivityType]] = None,
    token_account: Optional[str] = None,
    from_address: Optional[str] = None,
    to_address: Optional[str] = None,
    token: Optional[str] = None,
    amount: Optional[Tuple[float, float]] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    exclude_amount_zero: Optional[bool] = None,
) -> dict | str:
    """Export transfer history of an account (CSV or raw data)."""
    params: Dict[str, Any] = {"address": address}
    if activity_type is not None:
        params["activity_type"] = activity_type
    if token_account is not None:
        params["token_account"] = token_account
    if from_address is not None:
        params["from"] = from_address
    if to_address is not None:
        params["to"] = to_address
    if token is not None:
        params["token"] = token
    if amount is not None:
        params["amount"] = list(amount)
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if exclude_amount_zero is not None:
        params["exclude_amount_zero"] = exclude_amount_zero
    return fetch_solscan("/v2.0/account/transfer/export", params=params)


# Token APIs
def fetch_token_transfer(
    address: str,
    activity_type: Optional[List[TransferActivityType]] = None,
    from_address: Optional[str] = None,
    to_address: Optional[str] = None,
    amount: Optional[Tuple[float, float]] = None,
    block_time: Optional[Tuple[int, int]] = None,
    exclude_amount_zero: Optional[bool] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
) -> dict | str:
    """Get transfer data for a specific token (SPL asset), with optional filters."""
    params: Dict[str, Any] = {"address": address}
    if activity_type is not None:
        params["activity_type"] = activity_type
    if from_address is not None:
        params["from"] = from_address
    if to_address is not None:
        params["to"] = to_address
    if amount is not None:
        params["amount"] = list(amount)
    if block_time is not None:
        # block_time expects [start, end] Unix timestamps (in seconds)
        params["block_time"] = list(block_time)
    if exclude_amount_zero is not None:
        params["exclude_amount_zero"] = exclude_amount_zero
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/token/transfer", params=params)


def fetch_token_defi_activities(
    address: str,
    from_address: Optional[str] = None,
    platform: Optional[AddressList5] = None,
    source: Optional[AddressList5] = None,
    activity_type: Optional[List[DefiActivityType]] = None,
    token: Optional[str] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
) -> dict | str:
    """Get DeFi activities involving a specific token (e.g. swaps, liquidity events)."""
    params: Dict[str, Any] = {"address": address}
    if from_address is not None:
        params["from"] = from_address
    if platform is not None:
        params["platform"] = platform
    if source is not None:
        params["source"] = source
    if activity_type is not None:
        params["activity_type"] = activity_type
    if token is not None:
        params["token"] = token
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/token/defi/activities", params=params)


def fetch_token_meta(address: str) -> dict | str:
    """Get the on-chain metadata for a token (name, symbol, decimals, etc.)."""
    params = {"address": address}
    return fetch_solscan("/v2.0/token/meta", params=params)


def fetch_token_price(
    address: str,
    from_time: Optional[DateYYYYMMDD] = None,
    to_time: Optional[DateYYYYMMDD] = None,
) -> dict | str:
    """Get historical price data for a token (daily price points)."""
    params: Dict[str, Any] = {"address": address}
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    return fetch_solscan("/v2.0/token/price", params=params)


def fetch_token_holders(
    address: str,
    page: Optional[int] = None,
    page_size: Optional[PageSizeSmall] = None,
    from_amount: Optional[str] = None,
    to_amount: Optional[str] = None,
) -> dict | str:
    """Get the list of holders for a token (with optional holding amount filters)."""
    params: Dict[str, Any] = {"address": address}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if from_amount is not None:
        params["from_amount"] = from_amount  # expects numeric value as string
    if to_amount is not None:
        params["to_amount"] = to_amount  # expects numeric value as string
    return fetch_solscan("/v2.0/token/holders", params=params)


def fetch_token_list(
    sort_by: Optional[Literal["holder", "market_cap", "created_time"]] = None,
    sort_order: Optional[SortOrder] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
) -> dict | str:
    """Get a paginated list of tokens, optionally sorted by holders, market cap, or creation time."""
    params: Dict[str, Any] = {}
    if sort_by is not None:
        params["sort_by"] = sort_by
    if sort_order is not None:
        params["sort_order"] = sort_order
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/token/list", params=params)


def fetch_token_top() -> dict | str:
    """Get the list of top tokens (by market cap)."""
    # No query params; returns a fixed set of top tokens.
    return fetch_solscan("/v2.0/token/top")


def fetch_token_trending(limit: Optional[int] = None) -> dict | str:
    """Get the list of trending tokens (most searched or active)."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    return fetch_solscan("/v2.0/token/trending", params=params)


# NFT APIs
def fetch_new_nft(
    filter: Literal["created_time"],
    page: Optional[int] = None,
    page_size: Optional[PageSizeNft] = None,
) -> dict | str:
    """Get a list of newly created NFTs (sorted by creation time)."""
    params: Dict[str, Any] = {"filter": filter}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/nft/news", params=params)


def fetch_nft_activities(
    from_address: Optional[str] = None,
    to_address: Optional[str] = None,
    source: Optional[AddressList5] = None,
    activity_type: Optional[List[NftActivityType]] = None,
    from_time: Optional[int] = None,
    to_time: Optional[int] = None,
    token: Optional[str] = None,
    collection: Optional[str] = None,
    currency_token: Optional[str] = None,
    price: Optional[Tuple[float, float]] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
) -> dict | str:
    """Get NFT marketplace activities (sales, listings, bids, etc.), with various filters."""
    params: Dict[str, Any] = {}
    if from_address is not None:
        params["from"] = from_address
    if to_address is not None:
        params["to"] = to_address
    if source is not None:
        params["source"] = source
    if activity_type is not None:
        params["activity_type"] = activity_type
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if token is not None:
        params["token"] = token
    if collection is not None:
        params["collection"] = collection
    if currency_token is not None:
        params["currency_token"] = currency_token
    if price is not None:
        params["price"] = list(price)
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/nft/activities", params=params)


def fetch_nft_collection_lists(
    range: Optional[Literal[1, 7, 30]] = None,
    sort_order: Optional[SortOrder] = None,
    sort_by: Optional[Literal["items", "floor_price", "volumes"]] = None,
    page: Optional[int] = None,
    page_size: Optional[PageSizeCollection] = None,
    collection: Optional[str] = None,
) -> dict | str:
    """Get a list of NFT collections, with optional sorting and filtering."""
    params: Dict[str, Any] = {}
    if range is not None:
        params["range"] = range
    if sort_order is not None:
        params["sort_order"] = sort_order
    if sort_by is not None:
        params["sort_by"] = sort_by
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if collection is not None:
        params["collection"] = collection
    return fetch_solscan("/v2.0/nft/collection/lists", params=params)


def fetch_nft_collection_items(
    collection: str,
    sort_by: Optional[Literal["last_trade", "listing_price"]] = "last_trade",
    page: Optional[int] = 1,
    page_size: Optional[PageSizeNft] = 12,
) -> dict | str:
    """Get items (NFTs) in a specific collection, optionally sorted by last trade or listing price."""
    params: Dict[str, Any] = {"collection": collection}
    if sort_by is not None:
        params["sort_by"] = sort_by
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return fetch_solscan("/v2.0/nft/collection/items", params=params)


# Transaction APIs
def fetch_transaction_last(
    limit: Optional[PageSizeMedium] = None, filter: Optional[VoteFilter] = None
) -> dict | str:
    """Get the latest transactions across the chain (with optional vote-exclusion filter)."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if filter is not None:
        params["filter"] = filter
    return fetch_solscan("/v2.0/transaction/last", params=params)


def fetch_transaction_detail(tx: str) -> dict | str:
    """Get detailed parsed info of a transaction by signature."""
    params = {"tx": tx}
    return fetch_solscan("/v2.0/transaction/detail", params=params)


def fetch_transaction_actions(tx: str) -> dict | str:
    """Get high-level actions (transfers, swaps, NFT events) extracted from a transaction."""
    params = {"tx": tx}
    return fetch_solscan("/v2.0/transaction/actions", params=params)


# Block APIs
def fetch_last_block(limit: Optional[PageSizeMedium] = None) -> dict | str:
    """Get the latest blocks on the chain (summary info)."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    return fetch_solscan("/v2.0/block/last", params=params)


def fetch_block_transactions(
    block: int,
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
    exclude_vote: Optional[bool] = None,
    program: Optional[str] = None,
) -> dict | str:
    """Get transactions contained in a specific block (with optional filters)."""
    params: Dict[str, Any] = {"block": block}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if exclude_vote is not None:
        params["exclude_vote"] = exclude_vote
    if program is not None:
        params["program"] = program
    return fetch_solscan("/v2.0/block/transactions", params=params)


def fetch_block_detail(block: int) -> dict | str:
    """Get detailed information about a block by slot number."""
    params = {"block": block}
    return fetch_solscan("/v2.0/block/detail", params=params)


# Market APIs
def fetch_market_list(
    page: Optional[int] = None,
    page_size: Optional[PageSizeMedium] = None,
    program: Optional[str] = None,
) -> dict | str:
    """Get a list of newly listed pools/markets (optionally filtered by program)."""
    params: Dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if program is not None:
        params["program"] = program
    return fetch_solscan("/v2.0/market/list", params=params)


def fetch_market_info(address: str) -> dict | str:
    """Get market info for a given market (pool) address."""
    params = {"address": address}
    return fetch_solscan("/v2.0/market/info", params=params)


def fetch_market_volume(
    address: str, time: Optional[Tuple[int, int]] = None
) -> dict | str:
    """Get trading volume for a given market, optionally within a date range"""
    params = {"address": address}
    if time is not None:
        # 'time' expects [start_date, end_date] in YYYYMMDD format
        params["time"] = list(time)
    return fetch_solscan("/v2.0/market/volume", params=params)


# Monitoring API
def fetch_monitor_usage() -> dict | str:
    """Get the API usage and remaining compute units for the current API key (subscriber)."""
    return fetch_solscan("/v2.0/monitor/usage")


# Chain Information
def fetch_chain_information() -> dict | str:
    """Get overall Solana blockchain information (public endpoint)."""
    return fetch_solscan("", full_url="https://public-api.solscan.io/chaininfo")


def validate_address(address: str) -> dict:
    try:
        fetch_account_detail(address=address)
        return {"valid": True, "reason": "Address is valid"}
    except httpx.HTTPStatusError as e:
        return {"valid": False, "reason": e.response.json()["errors"]["message"]}
