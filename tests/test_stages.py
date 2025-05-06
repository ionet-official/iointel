import pytest

from iointel import register_custom_task, run_agents, Workflow, Agent
from iointel.src.utilities.decorators import _unregister_custom_task


@pytest.fixture
def custom_hi_task():
    @register_custom_task("hi")
    def execute_hi(task_metadata, text, agents, execution_metadata):
        return run_agents(
            objective=text,
            agents=agents,
            output_type=str,
        ).execute()

    yield
    _unregister_custom_task("hi")


@pytest.fixture
def wolfram_agent():
    return Agent(
        name="WolframAgent", instructions="You are an agent who is a super genius."
    )


@pytest.fixture
def greeter_agent():
    return Agent(
        name="GreeterAgent", instructions="You are an agent who is a walmart greeter."
    )


def test_custom_chainable(custom_hi_task, wolfram_agent):
    # Create a workflow using the new paradigm.
    tasks_list = Workflow(
        objective="what is my name", client_mode=False, agents=[wolfram_agent]
    )
    tasks_list.hi(
        objective="use your history to get my name and i want you greet me like a walmart greeter "
    )
    assert tasks_list.tasks


async def test_multistage_workflow(wolfram_agent, greeter_agent):
    tasks_list = Workflow(
        objective="what is my name", client_mode=False, agents=[wolfram_agent]
    )
    tasks_list.add_task(
        {
            "name": "multi_stage_demo",
            "text": "my name is cody",
            "task_metadata": {"client_mode": False},
            "execution_metadata": {
                "execution_mode": "parallel",
                "stages": [
                    {
                        "stage_type": "parallel",
                        "objective": "ask if the user wants fries with that..",
                        "context": {"greeting": "Hi"},
                        "result_type": "str",
                    },
                    {
                        "stage_type": "simple",
                        "objective": "Calculate 2+2.",
                        "context": {"calculation": "2+2"},
                        "result_type": "int",
                    },
                ],
            },
            "agents": [greeter_agent],
        }
    )
    result = await tasks_list.run_tasks()
    assert "_stage_2" in str(result["results"])
