from .helpers import AsyncLazyCaller
from ..task import Task
from prefect import task


async def _run_async(objective: str, **all_kwargs):
    context = all_kwargs.pop("context", None)
    result_type = all_kwargs.pop("result_type", None)
    agents = all_kwargs.pop("agents", None)
    result_validator = all_kwargs.pop("result_validator", None)
    result = await Task().a_run(
        objective=objective,
        context=context,
        result_type=result_type,
        agents=agents,
        result_validator=result_validator,
        **all_kwargs,
    )
    return result


@task(persist_result=False)
def run_agents(objective: str, **kwargs):
    """
    Synchronous lazy wrapper around Task().run.
    """
    return AsyncLazyCaller(_run_async, objective, **kwargs)
