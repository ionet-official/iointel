from iointel.src.agent_methods.tools.firecrawl import Crawler


async def test_crawl_the_page():
    crawler = Crawler()
    assert await crawler.async_scrape_url(url="https://firecrawl.dev/")
