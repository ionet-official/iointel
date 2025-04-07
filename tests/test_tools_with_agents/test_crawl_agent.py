import marvin

from iointel import Agent
from iointel.src.agent_methods.tools.firecrawl import Crawler


async def test_firecrawl():
    crawler = Crawler()
    agent = Agent(
        name="Agent",
        instructions="You are a crawler agent. Crawl web pages, retrieve information, do what user asks.",
    )
    result = await marvin.run_async(
        "Crawl this page: https://decrypt.co/306329/io-net-launches-generative-intelligence-platform-for-developers. "
        "What is the exact date of the io-intelligence first release? "
        "Provide the response in a format: dd-mm-yyyy",
        agents=[agent],
        tools=[crawler.async_scrape_url],
    )
    assert result is not None, "Expected a result from the agent run."
    assert "17-02-2025" in result
