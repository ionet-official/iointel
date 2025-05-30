import pytest
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from iointel.src.agent_methods.tools.duckduckgo import search_the_web


def test_duckduckgo_tool(monkeypatch):
    monkeypatch.setenv("DDGS_HTTP_PROXY", "http://127.0.0.1:7070")
    # wiremock/proxy must use httpv1
    monkeypatch.setenv("DDGS_HTTP_V1", "true")

    try:
        r = search_the_web("When did people fly to the moon?", max_results=3)
        assert r
    except DuckDuckGoSearchException as err:
        if "202 Ratelimit" in str(err):
            pytest.xfail(reason="DuckDuckGoSearchException - DDG rate limited us :(")
        raise
    except Exception:
        raise
