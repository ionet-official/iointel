import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Literal, Tuple, Annotated


import httpx
import pytest

# Replace these with your actual API key and wallet address
SOLSCAN_API_KEY = os.getenv('SOLSCAN_API_KEY')

# Base URL for Solscan Pro API V2
SOLSCAN_API_URL = "https://pro-api.solscan.io"


async def fetch_solscan(path, full_url: Optional[str] = None, params: Optional[dict] = None) -> dict | str:
    url = f"{SOLSCAN_API_URL}{path}"
    if full_url:
        url = full_url
    headers = {
        "token": SOLSCAN_API_KEY
    }
    async with httpx.AsyncClient() as client:
        print(f'Get {url=}, {params=}, {headers=}')
        response = await client.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return response.text

# Define common type aliases for readability and reuse
TransferActivityType = Literal["ACTIVITY_SPL_TRANSFER", "ACTIVITY_SPL_BURN",
                               "ACTIVITY_SPL_MINT", "ACTIVITY_SPL_CREATE_ACCOUNT"]
DefiActivityType = Literal["ACTIVITY_TOKEN_SWAP", "ACTIVITY_AGG_TOKEN_SWAP",
                            "ACTIVITY_TOKEN_ADD_LIQ", "ACTIVITY_TOKEN_REMOVE_LIQ",
                            "ACTIVITY_SPL_TOKEN_STAKE", "ACTIVITY_SPL_TOKEN_UNSTAKE",
                            "ACTIVITY_SPL_TOKEN_WITHDRAW_STAKE", "ACTIVITY_SPL_INIT_MINT"]
NftActivityType = Literal["ACTIVITY_NFT_SOLD", "ACTIVITY_NFT_LISTING",
                           "ACTIVITY_NFT_BIDDING", "ACTIVITY_NFT_CANCEL_BID",
                           "ACTIVITY_NFT_CANCEL_LIST", "ACTIVITY_NFT_REJECT_BID",
                           "ACTIVITY_NFT_UPDATE_PRICE", "ACTIVITY_NFT_LIST_AUCTION"]

PageSizeSmall = Literal[10, 20, 30, 40]                   # e.g., for smaller pagination sets&#8203;:contentReference[oaicite:0]{index=0}
PageSizeMedium = Literal[10, 20, 30, 40, 60, 100]         # e.g., for larger pagination sets&#8203;:contentReference[oaicite:1]{index=1}
PageSizeNft = Literal[12, 24, 36]                        # e.g., for NFT item listings&#8203;:contentReference[oaicite:2]{index=2}
PageSizeCollection = Literal[10, 18, 20, 30, 40]          # e.g., for collection lists&#8203;:contentReference[oaicite:3]{index=3}

SortOrder = Literal["asc", "desc"]                       # ascending or descending order&#8203;:contentReference[oaicite:4]{index=4}
TokenAccountType = Literal["token", "nft"]               # type of token account&#8203;:contentReference[oaicite:5]{index=5}
VoteFilter = Literal["exceptVote", "all"]                # filter to exclude vote transactions&#8203;:contentReference[oaicite:6]{index=6}
BoolStr = Literal["true", "false"]                       # 'true'/'false' string for certain flags&#8203;:contentReference[oaicite:7]{index=7}

AddressList5 = Annotated[List[str], "max_length 5"]      # up to 5 addresses (for filters like platform/source)&#8203;:contentReference[oaicite:8]{index=8}

DateYYYYMMDD = Annotated[int, "format YYYYMMDD"]         # date in YYYYMMDD format (e.g., 20240701)

# Account APIs
async def fetch_account_detail(address: str) -> dict | str:
    """Get the details of an account&#8203;:contentReference[oaicite:9]{index=9}."""
    params = {"address": address}
    return await fetch_solscan("/v2.0/account/detail", params=params)

async def fetch_account_transfer(address: str,
                           activity_type: Optional[List[TransferActivityType]] = None,
                           token_account: Optional[str] = None,
                           from_address: Optional[str] = None,
                           to_address: Optional[str] = None,
                           token: Optional[str] = None,
                           amount: Optional[Tuple[float, float]] = None,
                           from_time: Optional[int] = None,
                           to_time: Optional[int] = None,
                           exclude_amount_zero: Optional[bool] = None) -> dict | str:
    """Get transfer data of an account (with optional filters)&#8203;:contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}."""
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
        # Expect tuple (min, max) amount range&#8203;:contentReference[oaicite:12]{index=12}
        params["amount"] = list(amount)
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    if exclude_amount_zero is not None:
        params["exclude_amount_zero"] = exclude_amount_zero
    return await fetch_solscan("/v2.0/account/transfer", params=params)

async def fetch_account_defi_activities(address: str,
                                  activity_type: Optional[List[DefiActivityType]] = None,
                                  from_address: Optional[str] = None,
                                  platform: Optional[AddressList5] = None,
                                  source: Optional[AddressList5] = None,
                                  token: Optional[str] = None,
                                  from_time: Optional[int] = None,
                                  to_time: Optional[int] = None,
                                  page: Optional[int] = None,
                                  page_size: Optional[PageSizeMedium] = None) -> dict | str:
    """Get DeFi activities involving an account&#8203;:contentReference[oaicite:13]{index=13}."""
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
    return await fetch_solscan("/v2.0/account/defi/activities", params=params)

async def fetch_account_balance_change_activities(address: str,
                                            token_account: Optional[str] = None,
                                            token: Optional[str] = None,
                                            from_time: Optional[int] = None,
                                            to_time: Optional[int] = None,
                                            page_size: Optional[PageSizeMedium] = None,
                                            page: Optional[int] = None,
                                            remove_spam: Optional[BoolStr] = None,
                                            amount: Optional[Tuple[float, float]] = None,
                                            flow: Optional[Literal["in", "out"]] = None) -> dict | str:
    """Get balance change activities (token inflows/outflows) for an account&#8203;:contentReference[oaicite:14]{index=14}."""
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
        params["remove_spam"] = remove_spam  # expects "true" or "false" string&#8203;:contentReference[oaicite:15]{index=15}
    if amount is not None:
        params["amount"] = list(amount)
    if flow is not None:
        params["flow"] = flow
    return await fetch_solscan("/v2.0/account/balance_change", params=params)

async def fetch_account_transactions(address: str,
                               before: Optional[str] = None,
                               limit: Optional[Literal[10, 20, 30, 40]] = None) -> dict | str:
    """Get list of transactions for an account (with pagination)&#8203;:contentReference[oaicite:16]{index=16}&#8203;:contentReference[oaicite:17]{index=17}."""
    params: Dict[str, Any] = {"address": address}
    if before is not None:
        params["before"] = before
    if limit is not None:
        params["limit"] = limit
    return await fetch_solscan("/v2.0/account/transactions", params=params)

async def fetch_account_portfolio(address: str) -> dict | str:
    """Get the portfolio (token balances and values) for a given address&#8203;:contentReference[oaicite:18]{index=18}."""
    params = {"address": address}
    return await fetch_solscan("/v2.0/account/portfolio", params=params)

async def fetch_account_token_accounts(address: str,
                                 account_type: TokenAccountType,
                                 page: Optional[int] = None,
                                 page_size: Optional[PageSizeSmall] = None,
                                 hide_zero: Optional[bool] = None) -> dict | str:
    """Get token accounts of an address (either SPL tokens or NFTs)&#8203;:contentReference[oaicite:19]{index=19}&#8203;:contentReference[oaicite:20]{index=20}."""
    params: Dict[str, Any] = {"address": address, "type": account_type}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if hide_zero is not None:
        params["hide_zero"] = hide_zero
    return await fetch_solscan("/v2.0/account/token-accounts", params=params)

async def fetch_account_stake(address: str,
                        page: Optional[int] = None,
                        page_size: Optional[PageSizeSmall] = None) -> dict | str:
    """Get the list of stake accounts for a given wallet address&#8203;:contentReference[oaicite:21]{index=21}."""
    params: Dict[str, Any] = {"address": address}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await fetch_solscan("/v2.0/account/stake", params=params)

async def fetch_stake_rewards_export(address: str,
                               time_from: Optional[int] = None,
                               time_to: Optional[int] = None) -> dict | str:
    """Export staking reward history for an account (up to 5000 records)&#8203;:contentReference[oaicite:22]{index=22}&#8203;:contentReference[oaicite:23]{index=23}."""
    params: Dict[str, Any] = {"address": address}
    if time_from is not None:
        params["time_from"] = time_from
    if time_to is not None:
        params["time_to"] = time_to
    return await fetch_solscan("/v2.0/account/reward/export", params=params)

async def fetch_account_transfer_export(address: str,
                                  activity_type: Optional[List[TransferActivityType]] = None,
                                  token_account: Optional[str] = None,
                                  from_address: Optional[str] = None,
                                  to_address: Optional[str] = None,
                                  token: Optional[str] = None,
                                  amount: Optional[Tuple[float, float]] = None,
                                  from_time: Optional[int] = None,
                                  to_time: Optional[int] = None,
                                  exclude_amount_zero: Optional[bool] = None) -> dict | str:
    """Export transfer history of an account (CSV or raw data)&#8203;:contentReference[oaicite:24]{index=24}."""
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
    return await fetch_solscan("/v2.0/account/transfer/export", params=params)

# Token APIs
async def fetch_token_transfer(address: str,
                         activity_type: Optional[List[TransferActivityType]] = None,
                         from_address: Optional[str] = None,
                         to_address: Optional[str] = None,
                         amount: Optional[Tuple[float, float]] = None,
                         block_time: Optional[Tuple[int, int]] = None,
                         exclude_amount_zero: Optional[bool] = None,
                         page: Optional[int] = None,
                         page_size: Optional[PageSizeMedium] = None) -> dict | str:
    """Get transfer data for a specific token (SPL asset), with optional filters&#8203;:contentReference[oaicite:25]{index=25}&#8203;:contentReference[oaicite:26]{index=26}."""
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
        # block_time expects [start, end] Unix timestamps (in seconds)&#8203;:contentReference[oaicite:27]{index=27}
        params["block_time"] = list(block_time)
    if exclude_amount_zero is not None:
        params["exclude_amount_zero"] = exclude_amount_zero
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await fetch_solscan("/v2.0/token/transfer", params=params)

async def fetch_token_defi_activities(address: str,
                                from_address: Optional[str] = None,
                                platform: Optional[AddressList5] = None,
                                source: Optional[AddressList5] = None,
                                activity_type: Optional[List[DefiActivityType]] = None,
                                token: Optional[str] = None,
                                from_time: Optional[int] = None,
                                to_time: Optional[int] = None,
                                page: Optional[int] = None,
                                page_size: Optional[PageSizeMedium] = None) -> dict | str:
    """Get DeFi activities involving a specific token (e.g. swaps, liquidity events)&#8203;:contentReference[oaicite:28]{index=28}."""
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
    return await fetch_solscan("/v2.0/token/defi/activities", params=params)

async def fetch_token_meta(address: str) -> dict | str:
    """Get the on-chain metadata for a token (name, symbol, decimals, etc.)&#8203;:contentReference[oaicite:31]{index=31}."""
    params = {"address": address}
    return await fetch_solscan("/v2.0/token/meta", params=params)

async def fetch_token_price(address: str,
                      from_time: Optional[DateYYYYMMDD] = None,
                      to_time: Optional[DateYYYYMMDD] = None) -> dict | str:
    """Get historical price data for a token (daily price points)&#8203;:contentReference[oaicite:32]{index=32}&#8203;:contentReference[oaicite:33]{index=33}."""
    params: Dict[str, Any] = {"address": address}
    if from_time is not None:
        params["from_time"] = from_time
    if to_time is not None:
        params["to_time"] = to_time
    return await fetch_solscan("/v2.0/token/price", params=params)

async def fetch_token_holders(address: str,
                        page: Optional[int] = None,
                        page_size: Optional[PageSizeSmall] = None,
                        from_amount: Optional[str] = None,
                        to_amount: Optional[str] = None) -> dict | str:
    """Get the list of holders for a token (with optional holding amount filters)&#8203;:contentReference[oaicite:34]{index=34}."""
    params: Dict[str, Any] = {"address": address}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if from_amount is not None:
        params["from_amount"] = from_amount  # expects numeric value as string&#8203;:contentReference[oaicite:35]{index=35}
    if to_amount is not None:
        params["to_amount"] = to_amount      # expects numeric value as string&#8203;:contentReference[oaicite:36]{index=36}
    return await fetch_solscan("/v2.0/token/holders", params=params)

async def fetch_token_list(sort_by: Optional[Literal["holder", "market_cap", "created_time"]] = None,
                     sort_order: Optional[SortOrder] = None,
                     page: Optional[int] = None,
                     page_size: Optional[PageSizeMedium] = None) -> dict | str:
    """Get a paginated list of tokens, optionally sorted by holders, market cap, or creation time&#8203;:contentReference[oaicite:37]{index=37}&#8203;:contentReference[oaicite:38]{index=38}."""
    params: Dict[str, Any] = {}
    if sort_by is not None:
        params["sort_by"] = sort_by
    if sort_order is not None:
        params["sort_order"] = sort_order
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await fetch_solscan("/v2.0/token/list", params=params)

async def fetch_token_top() -> dict | str:
    """Get the list of top tokens (by market cap)&#8203;:contentReference[oaicite:39]{index=39}."""
    # No query params; returns a fixed set of top tokens.
    return await fetch_solscan("/v2.0/token/top")

async def fetch_token_trending(limit: Optional[int] = None) -> dict | str:
    """Get the list of trending tokens (most searched or active)&#8203;:contentReference[oaicite:40]{index=40}."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    return await fetch_solscan("/v2.0/token/trending", params=params)

# NFT APIs
async def fetch_new_nft(filter: Literal["created_time"],
                  page: Optional[int] = None,
                  page_size: Optional[PageSizeNft] = None) -> dict | str:
    """Get a list of newly created NFTs (sorted by creation time)&#8203;:contentReference[oaicite:41]{index=41}."""
    params: Dict[str, Any] = {"filter": filter}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await fetch_solscan("/v2.0/nft/news", params=params)

async def fetch_nft_activities(from_address: Optional[str] = None,
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
                         page_size: Optional[PageSizeMedium] = None) -> dict | str:
    """Get NFT marketplace activities (sales, listings, bids, etc.), with various filters&#8203;:contentReference[oaicite:42]{index=42}&#8203;:contentReference[oaicite:43]{index=43}."""
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
    return await fetch_solscan("/v2.0/nft/activities", params=params)

async def fetch_nft_collection_lists(range: Optional[Literal[1, 7, 30]] = None,
                               sort_order: Optional[SortOrder] = None,
                               sort_by: Optional[Literal["items", "floor_price", "volumes"]] = None,
                               page: Optional[int] = None,
                               page_size: Optional[PageSizeCollection] = None,
                               collection: Optional[str] = None) -> dict | str:
    """Get a list of NFT collections, with optional sorting and filtering&#8203;:contentReference[oaicite:44]{index=44}."""
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
    return await fetch_solscan("/v2.0/nft/collection/lists", params=params)

async def fetch_nft_collection_items(collection: str,
                               sort_by: Optional[Literal["last_trade", "listing_price"]] = "last_trade",
                               page: Optional[int] = 1,
                               page_size: Optional[PageSizeNft] = 12) -> dict | str:
    """Get items (NFTs) in a specific collection, optionally sorted by last trade or listing price&#8203;:contentReference[oaicite:45]{index=45}."""
    params: Dict[str, Any] = {"collection": collection}
    if sort_by is not None:
        params["sort_by"] = sort_by
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await fetch_solscan("/v2.0/nft/collection/items", params=params)

# Transaction APIs
async def fetch_transaction_last(limit: Optional[PageSizeMedium] = None,
                           filter: Optional[VoteFilter] = None) -> dict | str:
    """Get the latest transactions across the chain (with optional vote-exclusion filter)&#8203;:contentReference[oaicite:46]{index=46}&#8203;:contentReference[oaicite:47]{index=47}."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if filter is not None:
        params["filter"] = filter
    return await fetch_solscan("/v2.0/transaction/last", params=params)

async def fetch_transaction_detail(tx: str) -> dict | str:
    """Get detailed parsed info of a transaction by signature&#8203;:contentReference[oaicite:48]{index=48}."""
    params = {"tx": tx}
    return await fetch_solscan("/v2.0/transaction/detail", params=params)

async def fetch_transaction_actions(tx: str) -> dict | str:
    """Get high-level actions (transfers, swaps, NFT events) extracted from a transaction&#8203;:contentReference[oaicite:49]{index=49}."""
    params = {"tx": tx}
    return await fetch_solscan("/v2.0/transaction/actions", params=params)

# Block APIs
async def fetch_block_last(limit: Optional[PageSizeMedium] = None) -> dict | str:
    """Get the latest blocks on the chain (summary info)&#8203;:contentReference[oaicite:50]{index=50}."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    return await fetch_solscan("/v2.0/block/last", params=params)

async def fetch_block_transactions(block: int,
                             page: Optional[int] = None,
                             page_size: Optional[PageSizeMedium] = None,
                             exclude_vote: Optional[bool] = None,
                             program: Optional[str] = None) -> dict | str:
    """Get transactions contained in a specific block (with optional filters)&#8203;:contentReference[oaicite:51]{index=51}."""
    params: Dict[str, Any] = {"block": block}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if exclude_vote is not None:
        params["exclude_vote"] = exclude_vote
    if program is not None:
        params["program"] = program
    return await fetch_solscan("/v2.0/block/transactions", params=params)

async def fetch_block_detail(block: int) -> dict | str:
    """Get detailed information about a block by slot number&#8203;:contentReference[oaicite:52]{index=52}."""
    params = {"block": block}
    return await fetch_solscan("/v2.0/block/detail", params=params)

# Market APIs
async def fetch_market_list(page: Optional[int] = None,
                      page_size: Optional[PageSizeMedium] = None,
                      program: Optional[str] = None) -> dict | str:
    """Get a list of newly listed pools/markets (optionally filtered by program)&#8203;:contentReference[oaicite:53]{index=53}."""
    params: Dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if program is not None:
        params["program"] = program
    return await fetch_solscan("/v2.0/market/list", params=params)

async def fetch_market_info(address: str) -> dict | str:
    """Get market info for a given market (pool) address&#8203;:contentReference[oaicite:54]{index=54}."""
    params = {"address": address}
    return await fetch_solscan("/v2.0/market/info", params=params)

async def fetch_market_volume(address: str,
                        time: Optional[Tuple[int, int]] = None) -> dict | str:
    """Get trading volume for a given market, optionally within a date range&#8203;:contentReference[oaicite:55]{index=55}&#8203;:contentReference[oaicite:56]{index=56}."""
    params = {"address": address}
    if time is not None:
        # 'time' expects [start_date, end_date] in YYYYMMDD format&#8203;:contentReference[oaicite:57]{index=57}
        params["time"] = list(time)
    return await fetch_solscan("/v2.0/market/volume", params=params)

# Monitoring API
async def fetch_monitor_usage() -> dict | str:
    """Get the API usage and remaining compute units for the current API key (subscriber)&#8203;:contentReference[oaicite:58]{index=58}."""
    return await fetch_solscan("/v2.0/monitor/usage")

# Chain Information
async def fetch_chain_information() -> dict | str:
    """Get overall Solana blockchain information (public endpoint)&#8203;:contentReference[oaicite:59]{index=59}."""
    return await fetch_solscan("", full_url='https://public-api.solscan.io/chaininfo')

async def validate_address(address: str) -> str:
    try:
        detail = await fetch_account_detail(address=address)
        return 'Address is valid'
    except httpx.HTTPStatusError as e:
        return e.response.json()['errors']['message']


# Define the dictionary of functions, and an invocation example
tools_with_examples = {
    validate_address: {"address": "abc"},
    fetch_account_detail: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_transfer: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_defi_activities: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_balance_change_activities: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_transactions: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_portfolio: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_token_accounts: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ", "account_type": "token"},
    fetch_account_stake: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_stake_rewards_export: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_transfer_export: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_token_transfer: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_defi_activities: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_meta: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_price: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_holders: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_list: {},
    fetch_token_top: {},
    fetch_token_trending: {},
    fetch_new_nft: {"filter": "created_time"},
    fetch_nft_activities: {},
    fetch_nft_collection_lists: {},
    fetch_nft_collection_items: {"collection": "4P9XKtSJBscScF5NfM8h4V6yjRf2g1eG3U9w4X8hW8Z2"},
    fetch_transaction_last: {},
    fetch_transaction_detail: {"tx": "4QJaroEcVhbQYZoLeX2oXyTToaKcY6GoFSBQNMne6jdiMEQ6k8mWE8TMXhH7W2X1stdFFXb9Yb3Ly6ojFc6cMv2c"},
    fetch_transaction_actions: {"tx": "4QJaroEcVhbQYZoLeX2oXyTToaKcY6GoFSBQNMne6jdiMEQ6k8mWE8TMXhH7W2X1stdFFXb9Yb3Ly6ojFc6cMv2c"},
    fetch_block_last: {},
    fetch_block_transactions: {"block": 327993245},
    fetch_block_detail: {"block": 327993245},
    fetch_market_list: {},
    fetch_market_info: {"address": "EBHVuBXJrHQhxrXxduPPWTT9rRSrS42tLfU7eKi23sKE"},
    fetch_market_volume: {"address": "EBHVuBXJrHQhxrXxduPPWTT9rRSrS42tLfU7eKi23sKE"},
    fetch_monitor_usage: {},
    fetch_chain_information: {},
}


def test_all():
    for func, params in tools_with_examples.items():
        result = asyncio.run(func(**params))
        assert result # Make sure it returns something

def test_validate_address():
    assert 'Address is valid' in asyncio.run(validate_address(address='7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ'))
    assert 'Address [abc] is invalid' in asyncio.run(validate_address(address='abc'))
