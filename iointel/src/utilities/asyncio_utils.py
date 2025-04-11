import asyncio
import inspect
import threading
from typing import Coroutine, Any

_thread_pool_size = 10
_loops = [asyncio.new_event_loop() for _ in range(_thread_pool_size)]
_threads = [
    threading.Thread(target=_loops[i].run_forever, name="Async Runner", daemon=True)
    for i in range(_thread_pool_size)
]
_in_thread = [False] * _thread_pool_size


# This will block the calling thread until the coroutine is finished.
# Any exception that occurs in the coroutine is raised in the caller
def run_async(coro: Coroutine):
    global _loops, _threads, _in_thread
    available_thread_id = None
    for i in range(_thread_pool_size):
        if not _in_thread[i]:
            available_thread_id = i
            break
    if available_thread_id is None:
        raise Exception(
            "All 10 threads are in use. You are probably doing it wrong."
            "This function is supposed to run async functions from sync context."
        )
    _in_thread[available_thread_id] = True
    if not _threads[available_thread_id].is_alive():
        _threads[available_thread_id].start()
    future = asyncio.run_coroutine_threadsafe(coro, _loops[available_thread_id])
    result = future.result()
    _in_thread[available_thread_id] = False
    return result


async def await_if_needed(call_result: Any):
    if inspect.isawaitable(call_result):
        return await call_result
    else:
        return call_result
