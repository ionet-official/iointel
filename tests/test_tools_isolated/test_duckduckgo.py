from iointel.src.agent_methods.tools.duckduckgo import search_the_web_async


async def test_duckduckgo_tool():
    assert await search_the_web_async("When did people fly to the moon?", max_results=3)
