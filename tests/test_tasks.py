# tests/test_tasks.py
import pytest
from iointel.src.workflow import Workflow

def test_tasks_chain_basic():
    """
    Ensure that calling chainable methods appends tasks correctly.
    """
    f = Workflow(text="Sample text", client_mode=False)
    f.schedule_reminder(delay=10).council().sentiment()
    assert len(f.tasks) == 3

    assert f.tasks[0]["type"] == "schedule_reminder"
    assert f.tasks[0]["delay"] == 10
    assert f.tasks[1]["type"] == "council"
    assert f.tasks[2]["type"] == "sentiment"
    # We won't actually run tasks.run_tasks().
    # Instead, we just confirm the tasks are appended.

def test_tasks_custom():
    """
    Test that adding a custom step sets the correct fields.
    """
    flows = Workflow(text="Analyze this text", client_mode=True)
    flows.custom(
        name="my-custom-step",
        objective="Custom objective",
        instructions="Some instructions",
        my_extra="something"
    )
    assert len(flows.tasks) == 1
    c = flows.tasks[0]
    assert c["type"] == "custom"
    assert c["name"] == "my-custom-step"
    assert c["objective"] == "Custom objective"
    assert c["instructions"] == "Some instructions"
    assert c["kwargs"]["my_extra"] == "something"
