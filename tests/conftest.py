import pytest
import inspect
import functools
import asyncio


@pytest.hookspec(firstresult=True)
def pytest_timeout_set_timer(item: pytest.Item, settings):
    timeout = getattr(settings, "timeout", 0)
    if timeout <= 0:
        timeout = None
    if inspect.iscoroutinefunction(getattr(item, "obj", None)):
        # use own wrapper
        underlying = item.obj

        @functools.wraps(item.obj)
        async def timed_obj(*args, **kw):
            return await asyncio.wait_for(underlying(*args, **kw), timeout=timeout)

        item.obj = timed_obj
        return True
    return None  # use default
