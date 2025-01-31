import pytest

from framework.apis.client import client


def test_schedule_task(mock_requests):
    with pytest.raises(NotImplementedError):
        client.schedule_task("random junk")


def test_council_task(mock_requests):
    with pytest.raises(NotImplementedError):
        client.run_council_task("random junk")


@pytest.mark.xfail(reason="need to fix prompt", strict=True)
def test_reasoning_task(mock_requests):
    result = client.run_reasoning_task("random junk")
    assert result


def test_summarize_task(mock_requests):
    result = client.summarize_task("random junk")
    assert result


def test_sentiment(mock_requests):
    result = client.sentiment_analysis("random junk")
    assert result


@pytest.mark.xfail(reason="need to debug backend", strict=True)
def test_extract_entities(mock_requests):
    result = client.extract_entities("random junk")
    assert result


def test_translate_task(mock_requests):
    result = client.translate_text_task("random junk", target_language="spanish")
    assert result


def test_classify(mock_requests):
    result = client.classify_text("random junk", classify_by=["whatever"])
    assert result


def test_moderation_task(mock_requests):
    result = client.moderation_task("random junk")
    assert result


def test_custom_flow(mock_requests):
    result = client.custom_workflow(
        text="random junk",
        name="test flow",
        objective="do random junk",
        instructions="do whatever",
    )
    assert result


@pytest.mark.skip
def test_get_tools(mock_requests):
    result = client.get_tools()
    assert result


@pytest.mark.skip
def test_get_servers(mock_requests):
    result = client.get_servers()
    assert result


def test_get_agents(mock_requests):
    result = client.get_agents()
    assert result


def test_upload_workflow(mock_requests, tmp_path):
    tmpfile = tmp_path / "workflow.yml"
    tmpfile.write_text("""
name: whatever
""")
    result = client.upload_workflow_file(str(tmpfile))
    assert result
