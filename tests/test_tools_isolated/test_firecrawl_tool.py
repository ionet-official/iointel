from iointel.src.agent_methods.tools.firecrawl import Crawler

crawler = Crawler()

async def test_firecrawl():
    assert crawler.scrape_url(url="https://firecrawl.dev/")
