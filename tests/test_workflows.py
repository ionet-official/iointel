import pytest
from iointel.src.utilities.decorators import _unregister_custom_task
from iointel.src.utilities.constants import get_api_url, get_base_model, get_api_key
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from iointel import Agent, Workflow, register_custom_task, run_agents
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
    model_name=get_base_model(),
    provider=OpenAIProvider(base_url=get_api_url(), api_key=get_api_key()),
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
    assert float(results["sentiment"]) >= 0


async def test_defaulting_workflow():
    workflow = Workflow("Hello, how is your health today?", client_mode=False)
    workflow.translate_text(target_language="spanish").sentiment()
    results = (await workflow.run_tasks())["results"]
    assert "translate_text" in results, results
    assert float(results["sentiment"]) >= 0, results


async def test_translation_workflow(poet):
    workflow = Workflow(objective=text, agents=[poet], client_mode=False)
    results = (await workflow.translate_text(target_language="spanish").run_tasks())[
        "results"
    ]
    assert "galaxia" in results["translate_text"]


async def test_summarize_text_workflow(poet):
    workflow = Workflow(
        "This is a long text talking about nothing, emptiness and things like that. "
        "Nobody knows what it is about. The void gazes into you.",
        agents=[poet],
        client_mode=False,
    )
    results = (await workflow.summarize_text().run_tasks())["results"]
    assert (
        "emptiness" in results["summarize_text"].summary
        or "void" in results["summarize_text"].summary
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
    assert results["classify"] == "fact"


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
        for key, value in base.model_dump().items():
            if key == "model":
                # OpenAIModels cannot be compared by simple `==`, need more complex checks
                assert isinstance(base.model, OpenAIModel)
                assert isinstance(unpacked.model, OpenAIModel)
                assert base.model.model_name == unpacked.model.model_name
                assert base.model.base_url == unpacked.model.base_url
            else:
                assert getattr(unpacked, key) == value, (
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
