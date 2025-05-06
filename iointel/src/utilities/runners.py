from .helpers import LazyCaller
from ..task import Task
from ..agent_methods.data_models.datamodels import TaskDefinition


def _to_task_definition(
    objective: str, agents=None, conversation_id=None, **kwargs
) -> TaskDefinition:
    """
    Helper that merges the user’s provided fields into a TaskDefinition.
    If your code doesn’t revolve around TaskDefinition yet,
    you can keep this minimal.
    """
    return TaskDefinition(
        task_id=kwargs.get("task_id", "some-default"),
        name=kwargs.get("name", objective),
        objective=objective,
        agents=agents,
        task_metadata={"conversation_id": conversation_id},
        # put any other relevant fields here
        # text=kwargs.get("text"),
        # execution_metadata=kwargs.get("execution_metadata"),
    )


async def _run_stream(objective: str, **all_kwargs):
    definition = _to_task_definition(objective, **all_kwargs)
    output_type = all_kwargs.pop("output_type", None)

    agents = definition.agents or []
    return await Task(agents=agents).run_stream(
        definition=definition, output_type=output_type
    )


async def _run(objective: str, **all_kwargs):
    definition = _to_task_definition(objective, **all_kwargs)
    output_type = all_kwargs.pop("output_type", None)
    agents = definition.agents or []
    return await Task(agents=agents).run(
        definition=definition, output_type=output_type
    )


def run_agents_stream(objective: str, **kwargs) -> LazyCaller:
    """
    Asynchronous lazy wrapper around Task().run_stream.
    """
    return LazyCaller(_run_stream, objective, **kwargs)


# @task(persist_result=False)
async def run_agents(objective: str, **kwargs) -> LazyCaller:
    """
    Asynchronous lazy wrapper around Task().a_run.
    """
    return LazyCaller(_run, objective, **kwargs)
