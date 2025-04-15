from .helpers import LazyCaller
from ..task import Task
from prefect import task


def _run_agents(objective: str, **all_kwargs):
    return Task().run(
        objective=objective,
        **all_kwargs,
    )


@task(persist_result=False)
def run_agents(objective: str, **kwargs):
    """
    Synchronous lazy wrapper around Task().run.
    """
    return LazyCaller(_run_agents, objective, **kwargs)
