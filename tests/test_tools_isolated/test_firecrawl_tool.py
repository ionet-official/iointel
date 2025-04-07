import asyncio

from iointel.src.agent_methods.tools_before_rebase.firecrawl import Crawler


def test_crawl_the_page():
    crawler = Crawler()
    assert asyncio.run(crawler.async_scrape_url(url="https://firecrawl.dev/"))
