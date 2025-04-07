from iointel.src.agent_methods.tools.duckduckgo import search_the_web


def test_duckduckgo_tool():
    assert search_the_web("When did people fly to the moon?", max_results=3)
