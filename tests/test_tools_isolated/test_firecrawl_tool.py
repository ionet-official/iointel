from iointel.src.agent_methods.tools.firecrawl import Crawler


def test_crawl_the_page():
    crawler = Crawler()
    assert crawler.scrape_url(url="https://firecrawl.dev/")
