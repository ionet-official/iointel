import pytest
from fastapi.testclient import TestClient
from framework.apis.main import app  # or wherever your FastAPI app is defined

@pytest.fixture(scope="session")
def test_client():
    """Provide a session-scoped test client for all tests."""
    client = TestClient(app)
    return client