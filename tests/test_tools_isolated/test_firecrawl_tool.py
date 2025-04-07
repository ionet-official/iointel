import asyncio

from iointel.src.agent_methods.tools.firecrawl import Crawler


def test_crawl_the_page():
    crawler = Crawler()
    assert asyncio.run(crawler.async_scrape_url(url="https://firecrawl.dev/"))
