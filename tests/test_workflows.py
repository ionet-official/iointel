import pytest
from iointel.src.utilities.constants import get_api_url, get_base_model, get_api_key
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from iointel import Agent, Workflow
from iointel.src.agent_methods.data_models.datamodels import ModerationException

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

llm = OpenAIModel(
    model_name = get_base_model(),
    provider = OpenAIProvider(
                    base_url=get_api_url(),
                    api_key=get_api_key()
                )
    )

@pytest.fixture
def poet() -> Agent:
    agent = Agent(
        name="ArcanePoetAgent",
        instructions="You are an assistant specialized in arcane knowledge.",
        model=llm,
    )
    agent.id = "test-id"  # Temporary patch for the test fixture
    return agent

def test_composite_workflow(poet):
    workflow = Workflow(text=text, agents=[poet], client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()

    results = workflow.run_tasks()["results"]
    assert "translate_text" in results, results
    assert "sentiment" in results, results
    assert results["sentiment"] > 0

def test_defaulting_workflow():
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()
    results = workflow.run_tasks()["results"]
    assert "translate_text" in results, results
    assert results["sentiment"] >= 0, results


def test_translation_workflow(poet):
    workflow = Workflow(text=text, agents=[poet], client_mode=False)
    results = workflow.translate_text(target_language="spanish").run_tasks()["results"]
    assert "galaxia" in results["translate_text"]


def test_summarize_text_workflow(poet):
    workflow = Workflow(
        "This is a long text talking about nothing, emptiness and things like that. "
        "Nobody knows what it is about. The void gazes into you.",
        agents=[poet],
        client_mode=False,
    )
    results = workflow.summarize_text().run_tasks()["results"]
    assert (
        "emptiness" in results["summarize_text"].summary
        or "void" in results["summarize_text"].summary
    )


def test_solve_with_reasoning_workflow():
    workflow = Workflow("What's 2+2", client_mode=False)
    results = workflow.solve_with_reasoning().run_tasks()["results"]
    assert "4" in results["solve_with_reasoning"], results


def test_sentiment_workflow():
    # High sentiment = positive reaction
    workflow = Workflow("The dinner was awesome!", client_mode=False)
    results = workflow.sentiment().run_tasks()["results"]
    assert results["sentiment"] > 0.5, results


def test_extract_categorized_entities_workflow():
    workflow = Workflow("Alice and Bob are exchanging messages", client_mode=False)
    results = workflow.extract_categorized_entities().run_tasks()["results"]
    persons = results["extract_categorized_entities"]["persons"]
    assert "Alice" in persons and "Bob" in persons and len(persons) == 2, results


def test_classify_workflow():
    workflow = Workflow(
        "A major tech company has announced a breakthrough in battery technology",
        client_mode=False,
    )
    results = workflow.classify(
        classify_by=["fact", "fiction", "sci-fi", "fantasy"]
    ).run_tasks()["results"]
    assert results["classify"] == "fact"


def test_moderation_workflow():
    workflow = Workflow(
        "I absolutely hate this service! And i hate you! And all your friends!",
        client_mode=False,
    )
    with pytest.raises(ModerationException):
        workflow.moderation(threshold=0.25).run_tasks()["results"]


def test_custom_workflow():
    workflow = Workflow("Alice and Bob are exchanging messages", client_mode=False)
    results = workflow.custom(
        name="custom-task",
        objective="Give me names of the people in the text",
        instructions="Every name should be present in the result exactly once."
        "Format the result like this: Name1, Name2, ..., NameX",
    ).run_tasks()
    assert "Alice, Bob" in results["results"]["custom-task"], results

def test_task_level_agent_workflow(poet):
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(agents=[poet], target_language="spanish").sentiment()
    results = workflow.run_tasks()["results"]
    assert "translate_text" in results, results
    assert results["sentiment"] >= 0, results
