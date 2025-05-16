import uuid
import pytest

from iointel.src.agent_methods.tools.r2r import R2RClient
from iointel.src.utilities.constants import get_api_key, get_api_url


@pytest.fixture
def r2r_client() -> R2RClient:
    base_url = get_api_url().removesuffix("/api/v1") + "/api/r2r"
    return R2RClient(base_url, get_api_key())


@pytest.fixture
async def sample_document(r2r_client: R2RClient):
    doc_id = await r2r_client.create_document(
        content="Sample document so that database is not empty. Remember to remove it.",
        id=uuid.uuid4(),
    )
    yield doc_id
    await r2r_client.delete_document(doc_id)


async def test_list_documents(r2r_client: R2RClient):
    _, amount = await r2r_client.list_documents()
    assert amount >= 0, "Documents amount cannot be negative"


async def test_binary_document(r2r_client: R2RClient):
    doc_id = await r2r_client.create_document(
        content=b"Sample binary document so that database is not empty. Remember to remove it.",
        id=uuid.uuid4(),
    )
    try:
        infos, amount = await r2r_client.list_documents(ids=[doc_id])
        assert infos, "Expected non-empty document information"
        assert amount == 1, f"Expected exactly one document, found: {amount}"
    finally:
        await r2r_client.delete_document(doc_id)


async def test_rag_search(r2r_client: R2RClient, sample_document: uuid.UUID):
    result = await r2r_client.rag_search("What do you know about samples?")
    assert result, "Expected non-empty RAG result"
    assert result["answer"], "Expected non-empty RAG answer"
    assert (
        "database" in result["answer"]
    ), "Sample document has to be referenced in RAG answer"
    assert result["citations"], "Expected to see at least one citation"
