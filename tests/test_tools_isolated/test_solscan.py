import os

import pytest

from iointel.src.agent_methods.tools.solscan import (
    validate_address,
    fetch_account_detail,
    fetch_account_transfer,
    fetch_account_defi_activities,
    fetch_account_balance_change_activities,
    fetch_account_transactions,
    fetch_account_portfolio,
    fetch_account_token_accounts,
    fetch_account_stake,
    fetch_stake_rewards_export,
    fetch_account_transfer_export,
    fetch_token_transfer,
    fetch_token_defi_activities,
    fetch_token_meta,
    fetch_token_price,
    fetch_token_holders,
    fetch_token_list,
    fetch_token_top,
    fetch_token_trending,
    fetch_new_nft,
    fetch_nft_activities,
    fetch_nft_collection_lists,
    fetch_nft_collection_items,
    fetch_transaction_last,
    fetch_transaction_detail,
    fetch_transaction_actions,
    fetch_last_block,
    fetch_block_transactions,
    fetch_block_detail,
    fetch_market_list,
    fetch_market_info,
    fetch_market_volume,
    fetch_monitor_usage,
    fetch_chain_information,
)

# Define the dictionary of functions, and an invocation example
tools_with_examples = {
    validate_address: {"address": "abc"},
    fetch_account_detail: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_transfer: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_account_defi_activities: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_account_balance_change_activities: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_account_transactions: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_account_portfolio: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_account_token_accounts: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ",
        "account_type": "token",
    },
    fetch_account_stake: {"address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"},
    fetch_stake_rewards_export: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_account_transfer_export: {
        "address": "7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    },
    fetch_token_transfer: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_defi_activities: {
        "address": "So11111111111111111111111111111111111111112"
    },
    fetch_token_meta: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_price: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_holders: {"address": "So11111111111111111111111111111111111111112"},
    fetch_token_list: {},
    fetch_token_top: {},
    fetch_token_trending: {},
    fetch_new_nft: {"filter": "created_time"},
    fetch_nft_activities: {},
    fetch_nft_collection_lists: {},
    fetch_nft_collection_items: {
        "collection": "4P9XKtSJBscScF5NfM8h4V6yjRf2g1eG3U9w4X8hW8Z2"
    },
    fetch_transaction_last: {},
    fetch_transaction_detail: {
        "tx": "4QJaroEcVhbQYZoLeX2oXyTToaKcY6GoFSBQNMne6jdiMEQ6k8mWE8TMXhH7W2X1stdFFXb9Yb3Ly6ojFc6cMv2c"
    },
    fetch_transaction_actions: {
        "tx": "4QJaroEcVhbQYZoLeX2oXyTToaKcY6GoFSBQNMne6jdiMEQ6k8mWE8TMXhH7W2X1stdFFXb9Yb3Ly6ojFc6cMv2c"
    },
    fetch_last_block: {},
    fetch_block_transactions: {"block": 327993245},
    fetch_block_detail: {"block": 327993245},
    fetch_market_list: {},
    fetch_market_info: {"address": "EBHVuBXJrHQhxrXxduPPWTT9rRSrS42tLfU7eKi23sKE"},
    fetch_market_volume: {"address": "EBHVuBXJrHQhxrXxduPPWTT9rRSrS42tLfU7eKi23sKE"},
    fetch_monitor_usage: {},
    fetch_chain_information: {},
}


@pytest.mark.skipif(
    os.getenv("CI") is not None,
    reason="Don't test all of them in CI, fails because of the limits",
)
@pytest.mark.parametrize("tool,params", tools_with_examples.items())
async def test_all_endpoints(tool, params):
    result = await tool(**params)
    assert result  # Make sure it returns something


async def test_validate_address():
    valid = await validate_address(
        address="7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ"
    )
    invalid = await validate_address(address="abc")
    assert valid["valid"] and "Address is valid" in valid["reason"]
    assert not invalid["valid"] and "Address [abc] is invalid" in invalid["reason"]
