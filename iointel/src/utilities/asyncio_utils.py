import asyncio
import inspect
import threading
from typing import Coroutine, Any

_loop = asyncio.new_event_loop()

_thr = threading.Thread(target=_loop.run_forever, name="Async Runner", daemon=True)


# This will block the calling thread until the coroutine is finished.
# Any exception that occurs in the coroutine is raised in the caller
def run_async(coro: Coroutine):
    if not _thr.is_alive():
        _thr.start()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


async def await_if_needed(call_result: Any):
    if inspect.isawaitable(call_result):
        await call_result
    else:
        return call_result
