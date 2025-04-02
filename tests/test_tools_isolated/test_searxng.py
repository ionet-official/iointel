from iointel.src.agent_methods.tools_before_rebase.searxng import SearxngClient


# Run searxng locally first
def test_searxng_tool():
    client = SearxngClient(base_url='http://localhost:8080')
    assert client.search_sync('When did people fly to the moon?')
