from .asyncio_utils import run_async
from .helpers import LazyCaller, AsyncLazyCaller
from ..task import Task
from prefect import task


async def _run_async(objective: str, **all_kwargs):
    context = all_kwargs.pop("context", None)
    result_type = all_kwargs.pop("result_type", None)
    agents = all_kwargs.pop("agents", None)
    result_validator = all_kwargs.pop("result_validator", None)
    return Task().a_run(
        objective=objective,
        context=context,
        result_type=result_type,
        agents=agents,
        result_validator=result_validator,
        **all_kwargs,
    )


def _run_sync(objective: str, **all_kwargs):
    return run_async(_run_async(objective, **all_kwargs))


@task(persist_result=False)
def run_agents_async(objective: str, **kwargs) -> AsyncLazyCaller:
    """
    Asynchronous lazy wrapper around Task().a_run.
    """
    return AsyncLazyCaller(_run_async, objective, **kwargs)


@task(persist_result=False)
def run_agents(objective: str, **kwargs) -> LazyCaller:
    """
    Synchronous lazy wrapper around Task().run.
    """
    return LazyCaller(_run_sync, objective, **kwargs)
