import pytest
import requests
import os

from fastapi.testclient import TestClient

INTERNAL_SERVER = "http://testserver"
os.environ["BASE_URL"] = os.environ.get("BASE_URL", INTERNAL_SERVER)

@pytest.fixture(scope="session")
def test_client():
    """Provide a session-scoped test client for all tests."""
    from framework.apis.agent_main import app  # or wherever your FastAPI app is defined
    client = TestClient(app)
    return client

@pytest.fixture(scope="module")
def mock_requests(test_client, module_mocker):
    def make_replacement(method, orig_request=requests.request):
        def patched(url, *args, **kw):
            if INTERNAL_SERVER in str(url):
                return test_client.request(method, url, *args, **kw)
            return orig_request(method, url, *args, **kw)
        return patched

    module_mocker.patch("requests.post", new=make_replacement('POST'))
    module_mocker.patch("requests.get", new=make_replacement('GET'))
    yield
    module_mocker.resetall()
