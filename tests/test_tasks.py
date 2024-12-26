# tests/test_tasks.py
import pytest
from framework.src.agents import Tasks

def test_tasks_chain_basic():
    """
    Ensure that calling chainable methods appends tasks correctly.
    """
    t = Tasks(text="Sample text", client_mode=False)
    t.schedule_reminder(delay=10).council().sentiment()
    assert len(t.tasks) == 3

    assert t.tasks[0]["type"] == "schedule_reminder"
    assert t.tasks[0]["delay"] == 10
    assert t.tasks[1]["type"] == "council"
    assert t.tasks[2]["type"] == "sentiment"
    # We won't actually run tasks.run_tasks().
    # Instead, we just confirm the tasks are appended.

def test_tasks_custom():
    """
    Test that adding a custom step sets the correct fields.
    """
    tasks = Tasks(text="Analyze this text", client_mode=True)
    tasks.custom(
        name="my-custom-step",
        objective="Custom objective",
        instructions="Some instructions",
        my_extra="something"
    )
    assert len(tasks.tasks) == 1
    c = tasks.tasks[0]
    assert c["type"] == "custom"
    assert c["name"] == "my-custom-step"
    assert c["objective"] == "Custom objective"
    assert c["instructions"] == "Some instructions"
    assert c["kwargs"]["my_extra"] == "something"