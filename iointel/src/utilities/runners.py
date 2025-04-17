from .helpers import LazyCaller
from ..task import Task
from prefect import task


@task(persist_result=False)
def run_agents(objective: str, **kwargs):
    """
    Synchronous lazy wrapper around Task().run.
    """
    return LazyCaller(Task().run, objective=objective, **kwargs)
