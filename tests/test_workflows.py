import pytest

from iointel import Agent, Workflow

text = """A long time ago, In a galaxy far, far away, 
It is a period of civil wars in the galaxy. 
A brave alliance of underground freedom fighters has challenged the tyranny and oppression of the awesome GALACTIC EMPIRE.
Striking from a fortress hidden among the billion stars of the galaxy, 
rebel spaceships have won their first victory in a battle with the powerful Imperial Starfleet. 
The EMPIRE fears that another defeat could bring a thousand more solar systems into the rebellion, 
and Imperial control over the galaxy would be lost forever.
To crush the rebellion once and for all, the EMPIRE is constructing a sinister new battle station. 
Powerful enough to destroy an entire planet, its completion spells certain doom for the champions of freedom.
"""


@pytest.fixture
def poet() -> Agent:
    return Agent(
        name="ArcanePoetAgent",
        instructions="You are an assistant specialized in arcane knowledge.",
    )


def test_composite_workflow(poet):
    workflow = Workflow(text=text, agents=[poet], client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()

    results = workflow.run_tasks()["results"]
    assert "translate_text" in results, results
    assert "sentiment" in results, results


def test_defaulting_workflow():
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()
    results = workflow.run_tasks()["results"]
    assert "translate_text" in results, results
    assert results["sentiment"] > 0, results
