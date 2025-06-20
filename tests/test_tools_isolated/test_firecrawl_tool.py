import os

import pytest
from iointel.src.agent_methods.tools.firecrawl import Crawler


@pytest.mark.skipif(
    not os.environ.get("FIRECRAWL_API_KEY"),
    reason="FIRECRAWL_API_KEY must be set to run this test",
)
async def test_firecrawl():
    crawler = Crawler()
    assert crawler.scrape_url(url="https://firecrawl.dev/")
