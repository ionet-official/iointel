from .helpers import AsyncLazyCaller
from ..task import Task
from prefect import task


async def _run_async(objective: str, **all_kwargs):
    return await Task().a_run(
        objective=objective,
        **all_kwargs,
    )


@task(persist_result=False)
def run_agents(objective: str, **kwargs):
    """
    Synchronous lazy wrapper around Task().run.
    """
    return AsyncLazyCaller(_run_async, objective, **kwargs)
