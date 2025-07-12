import os
import pytest
from iointel.src.utilities.decorators import _unregister_custom_task
from iointel.src.utilities.constants import get_api_url, get_base_model, get_api_key
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from iointel import Agent, Workflow, register_custom_task, run_agents
from iointel.src.agent_methods.data_models.datamodels import (
    ModerationException,
    PersonaConfig,
)


def extract_result_value(result):
    """Helper to extract the actual result value from the full result structure."""
    return result.get("result", result) if isinstance(result, dict) else result

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

# Use OpenAI models for tests to avoid API key issues
llm = OpenAIModel(
    model_name="gpt-4o",
    provider=OpenAIProvider(
        base_url="https://api.openai.com/v1", 
        api_key=os.getenv("OPENAI_API_KEY")
    ),
)


@pytest.fixture
def custom_hi_task():
    @register_custom_task("hi")
    def execute_hi(task_metadata, objective, agents, execution_metadata):
        return run_agents(
            objective=objective,
            agents=agents,
            output_type=str,
        ).execute()

    yield
    _unregister_custom_task("hi")


@pytest.fixture
def poet() -> Agent:
    agent = Agent(
        persona=PersonaConfig(name="garbage guy", bio="arcane janitor"),
        name="ArcanePoetAgent",
        instructions="You are an assistant specialized in arcane knowledge.",
        model=llm,
    )
    return agent


async def test_composite_workflow(poet):
    workflow = Workflow(objective=text, agents=[poet], client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()

    results = (await workflow.run_tasks())["results"]
    assert "translate_text" in results, results
    assert "sentiment" in results, results
    # Extract result value from the full result structure
    sentiment_result = results["sentiment"]
    sentiment_value = sentiment_result.get("result", sentiment_result) if isinstance(sentiment_result, dict) else sentiment_result
    assert float(sentiment_value) >= 0


async def test_defaulting_workflow():
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()
    results = (await workflow.run_tasks())["results"]
    assert "translate_text" in results, results
    # Extract result value from the full result structure
    sentiment_result = results["sentiment"]
    sentiment_value = sentiment_result.get("result", sentiment_result) if isinstance(sentiment_result, dict) else sentiment_result
    assert float(sentiment_value) >= 0, results


async def test_translation_workflow(poet):
    workflow = Workflow(objective=text, agents=[poet], client_mode=False)
    results = (await workflow.translate_text(target_language="spanish").run_tasks())[
        "results"
    ]
    # Extract result value from the full result structure
    translate_result = results["translate_text"]
    translate_value = translate_result.get("result", translate_result) if isinstance(translate_result, dict) else translate_result
    assert "galaxia" in translate_value


async def test_summarize_text_workflow(poet):
    workflow = Workflow(
        "This is a long text talking about nothing, emptiness and things like that. "
        "Nobody knows what it is about. The void gazes into you.",
        agents=[poet],
        client_mode=False,
    )
    results = (await workflow.summarize_text().run_tasks())["results"]
    # Extract result value from the full result structure
    summarize_result = results["summarize_text"]
    summarize_value = summarize_result.get("result", summarize_result) if isinstance(summarize_result, dict) else summarize_result
    summary_text = summarize_value.summary if hasattr(summarize_value, 'summary') else str(summarize_value)
    assert (
        "emptiness" in summary_text
        or "void" in summary_text
    )


@pytest.mark.skip(reason="Reasoning is prone to looping forever")
async def test_solve_with_reasoning_workflow():
    workflow = Workflow("What's 2+2", client_mode=False)
    results = (await workflow.solve_with_reasoning().run_tasks())["results"]
    assert "4" in results["solve_with_reasoning"], results


async def test_sentiment_workflow():
    # High sentiment = positive reaction
    workflow = Workflow("The dinner was awesome!", client_mode=False)
    results = (await workflow.sentiment().run_tasks())["results"]
    assert float(results["sentiment"]) > 0.5, results


async def test_extract_categorized_entities_workflow():
    workflow = Workflow("Alice and Bob are exchanging messages", client_mode=False)
    results = (await workflow.extract_categorized_entities().run_tasks())["results"]
    persons = results["extract_categorized_entities"]["persons"]
    assert "Alice" in persons and "Bob" in persons and len(persons) == 2, results


async def test_classify_workflow():
    workflow = Workflow(
        "A major tech company has announced a breakthrough in battery technology",
        client_mode=False,
    )
    results = (
        await workflow.classify(
            classify_by=["fact", "fiction", "sci-fi", "fantasy"]
        ).run_tasks()
    )["results"]
    assert results["classify"] == "fact"


async def test_moderation_workflow():
    workflow = Workflow(
        "I absolutely hate this service! And i hate you! And all your friends!",
        client_mode=False,
    )
    with pytest.raises(ModerationException):
        (await workflow.moderation(threshold=0.25).run_tasks())["results"]


async def test_custom_workflow():
    workflow = Workflow("Alice and Bob are exchanging messages", client_mode=False)
    results = await workflow.custom(
        name="custom-task",
        objective="""Give me names of the people in the text.
            Every name should be present in the result exactly once.
            Format the result like this: Name1, Name2, ..., NameX""",
    ).run_tasks()
    assert "Alice, Bob" in results["results"]["custom-task"], results


async def test_task_level_agent_workflow(poet):
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(agents=[poet], target_language="spanish").sentiment()
    results = (await workflow.run_tasks())["results"]
    assert "translate_text" in results, results
    assert float(results["sentiment"]) >= 0, results


async def test_sentiment_classify_workflow():
    workflow = Workflow(
        "A major tech company has announced a breakthrough in battery technology",
        client_mode=False,
    )
    results = (
        await workflow.classify(
            classify_by=["fact", "fiction", "sci-fi", "fantasy"]
        ).run_tasks()
    )["results"]
    # Extract result value from the full result structure
    classify_result = results["classify"]
    classify_value = extract_result_value(classify_result)
    assert classify_value == "fact"


async def test_custom_steps_workflow(custom_hi_task, poet):
    workflow = Workflow("Goku has a power level of over 9000", client_mode=False)
    results = (await workflow.hi(agents=[poet]).run_tasks())["results"]
    assert any(
        phrase in results["hi"].lower()
        for phrase in ["over 9000", "Goku", "9000", "power level", "over 9000!"]
    ), f"Unexpected result: {results['hi']}"


def _ensure_agents_equal(
    left: list[Agent] | None, right: list[Agent] | None, check_api_key: bool
):
    assert len(left or ()) == len(right or ()), (
        "Expected roundtrip to retain agent amount"
    )
    for base, unpacked in zip(left or (), right or ()):
        if not check_api_key:
            base = base.model_copy(update={"api_key": ""})
            unpacked = unpacked.model_copy(update={"api_key": ""})
        for key in base.model_dump():
            if key == "model":
                # OpenAIModels cannot be compared by simple `==`, need more complex checks
                assert isinstance(base.model, OpenAIModel)
                assert isinstance(unpacked.model, OpenAIModel)
                assert base.model.model_name == unpacked.model.model_name
                assert base.model.base_url == unpacked.model.base_url
            else:
                assert getattr(unpacked, key) == getattr(base, key), (
                    "Expected roundtrip to retain agent parameters"
                )


@pytest.mark.parametrize("store_creds", [True, False])
def test_yaml_roundtrip(custom_hi_task, poet, store_creds: bool):
    wf_base: Workflow = Workflow(
        "Goku has a power level of over 9000", client_mode=False
    ).hi(agents=[poet])
    yml = wf_base.to_yaml("test_workflow", store_creds=store_creds)
    wf_unpacked = Workflow.from_yaml(yml)
    assert "Goku" in yml

    _ensure_agents_equal(wf_base.agents, wf_unpacked.agents, store_creds)
    assert wf_base.objective == wf_unpacked.objective, (
        "Expected roundtrip to retain objective"
    )
    assert wf_base.client_mode == wf_unpacked.client_mode, (
        "Expected roundtrip to retain client_mode"
    )
    assert len(wf_base.tasks) == len(wf_unpacked.tasks), (
        "Expected roundtrip to retain task amount"
    )
    for base, unpacked in zip(wf_base.tasks, wf_unpacked.tasks):
        base_noagent = dict(base, agents=None)
        unpacked_noagent = dict(unpacked, agents=None)
        for key, value in base_noagent.items():
            assert unpacked_noagent[key] == value, (
                "Expected roundtrip to retain task info"
            )
        _ensure_agents_equal(base["agents"], unpacked["agents"], store_creds)
