import asyncio

import pytest

from iointel.src.agent_methods.tools_before_rebase.solscan import (
    tools_with_examples,
    validate_address,
)


@pytest.mark.skip(reason="Don't test all of them in CI, fails because of the limits")
def test_all_endpoints():
    for func, params in tools_with_examples.items():
        result = asyncio.run(func(**params))
        assert result  # Make sure it returns something


def test_validate_address():
    assert "Address is valid" in asyncio.run(
        validate_address(address="7ZjHeeYEesmBs4N6aDvCQimKdtJX2bs5boXpJmpG2bZJ")
    )
    assert "Address [abc] is invalid" in asyncio.run(validate_address(address="abc"))
