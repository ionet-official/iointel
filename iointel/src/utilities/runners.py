from typing import Sequence, Dict, Any

from .helpers import LazyCaller
from ..task import Task
from ..agent_methods.data_models.datamodels import TaskDefinition
from ..agent_methods.agents.agents_factory import agent_or_swarm


def _to_task_definition(
    objective: str,
    agents=None,
    conversation_id=None,
    name=None,
    task_id="some_default",
    context=None,
    **kwargs,
) -> TaskDefinition:
    """
    Helper that merges the user's provided fields into a TaskDefinition.
    If your code doesn't revolve around TaskDefinition yet,
    you can keep this minimal.
    """
    if isinstance(agents, Sequence):
        agents = agent_or_swarm(agents, store_creds=True)
    return TaskDefinition(
        task_id=task_id,
        name=name or objective,
        objective=objective,
        agents=agents,
        task_metadata={
            "conversation_id": conversation_id,
            "context": context,
        },
        # put any other relevant fields here
        # text=kwargs.get("text"),
        # execution_metadata=kwargs.get("execution_metadata"),
    )


async def _run_stream(objective: str, output_type=None, **all_kwargs):
    definition = _to_task_definition(objective, **all_kwargs)
    agents = definition.agents or []
    return await Task(agents=agents).run_stream(
        definition=definition, output_type=output_type
    )


async def _run(objective: str, output_type=None, **all_kwargs):
    definition = _to_task_definition(objective, **all_kwargs)
    agents = definition.agents or []
    return await Task(agents=agents).run(definition=definition, output_type=output_type)


async def _unpack(func, *args, **kwargs) -> Dict[str, Any]:
    result = await (await func(*args, **kwargs)).execute()
    # Return full result structure to preserve tool_usage_results
    return result


def run_agents_stream(objective: str, **kwargs) -> LazyCaller:
    """
    Asynchronous lazy wrapper around Task().run_stream.
    """
    return LazyCaller(_unpack, _run_stream, objective, **kwargs)


# @task(persist_result=False)
def run_agents(objective: str, **kwargs) -> LazyCaller:
    """
    Asynchronous lazy wrapper around Task().run.
    """
    return LazyCaller(_unpack, _run, objective, **kwargs)
