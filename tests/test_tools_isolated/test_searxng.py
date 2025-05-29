import os

import pytest

from iointel.src.agent_methods.tools.searxng import SearxngClient


# Run searxng locally first
@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Coudn't run searxng in github CI"
)
def test_searxng_tool():
    client = SearxngClient(base_url="http://localhost:8080")
    assert client.search("When did people fly to the moon?")
