import pytest
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from iointel.src.agent_methods.tools.duckduckgo import search_the_web


def test_duckduckgo_tool():
    try:
        assert search_the_web("When did people fly to the moon?", max_results=3)
    except DuckDuckGoSearchException as err:
        if "202 Ratelimit" in str(err):
            pytest.xfail(reason="DDG rate limited us :(")
        raise
