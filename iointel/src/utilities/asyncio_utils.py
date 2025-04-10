import asyncio
import inspect
import threading
from typing import Coroutine, Any

_loop = asyncio.new_event_loop()
_thr = threading.Thread(target=_loop.run_forever, name="Async Runner", daemon=True)
_in_thread = False


# This will block the calling thread until the coroutine is finished.
# Any exception that occurs in the coroutine is raised in the caller
def run_async(coro: Coroutine):
    if not _thr.is_alive():
        _thr.start()
    global _in_thread
    if _in_thread:
        raise Exception("You are already inside run_async thread. It's a loop!")
    _in_thread = True
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    result = future.result()
    _in_thread = False
    return result


async def await_if_needed(call_result: Any):
    if inspect.isawaitable(call_result):
        return await call_result
    else:
        return call_result
