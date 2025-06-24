import pytest

from iointel.client import client


def test_reasoning_task():
    result = client.run_reasoning_task("I need to add 2 and 2")
    assert result


def test_summarize_task():
    result = client.summarize_task(
        "This is a long text talking about nothing, emptiness and things like that. Nobody knows what it is about. The void gazes into you."
    )
    assert result
    result = client.summarize_task(
        "Breaking news: local sports team wins!", max_words=50
    )
    assert result


def test_sentiment():
    result = client.sentiment_analysis("random junk")
    assert result


def test_extract_entities():
    result = client.extract_entities("random junk")
    assert result


def test_translate_task():
    result = client.translate_text_task("random junk", target_language="spanish")
    assert result


def test_classify():
    result = client.classify_text("random junk", classify_by=["whatever"])
    assert result


def test_moderation_task():
    result = client.moderation_task("random junk")
    assert result
    result = client.moderation_task("random junk", threshold=1.0)
    assert result


def test_custom_flow():
    result = client.custom_workflow(
        text="random junk",
        name="test flow",
        objective="do random junk",
        instructions="do whatever",
    )
    assert result


@pytest.mark.skip
def test_get_tools():
    result = client.get_tools()
    assert result


@pytest.mark.skip
def test_get_servers():
    result = client.get_servers()
    assert result


def test_get_agents():
    result = client.get_agents()
    assert result


def test_upload_workflow(tmp_path):
    tmpfile = tmp_path / "workflow.yml"
    tmpfile.write_text("""
name: whatever
""")
    with pytest.raises(NotImplementedError):
        client.upload_workflow_file(str(tmpfile))
