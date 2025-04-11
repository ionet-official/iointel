import asyncio
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, model_serializer

import logging
import os

from iointel.src.utilities.asyncio_utils import await_if_needed, run_async


def make_logger(name: str, level: str = "INFO"):
    logger = logging.getLogger(name)
    level_name = os.environ.get("AGENT_LOGGING_LEVEL", level).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(numeric_level)
    return logger


logger = make_logger(__name__)


class AsyncLazyCaller(BaseModel):
    func: Callable
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    _evaluated: bool = False
    _result: Any = None
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(
            func=func,
            args=args,
            kwargs=kwargs,
            name=kwargs.get("name") or func.__name__,
        )
        logger.debug(f"CREATE NEW CALLER with {kwargs}")
        self._evaluated = False
        self._result = None

    async def _resolve_nested(self, value: Any) -> Any:
        logger.debug("Resolving: %s", value)
        if hasattr(value, "execute_async") and callable(value.execute_async):
            logger.debug("Resolving lazy object: %s", value)
            result = await value.execute_async()
            resolved = await self._resolve_nested(result)
            logger.debug("Resolved lazy object to: %s", resolved)
            return resolved
        elif isinstance(value, dict):
            logger.debug("Resolving dict: %s", value)
            resolved_values = await asyncio.gather(
                *(self._resolve_nested(v) for v in value.values())
            )
            return {k: r for k, r in zip(value.keys(), resolved_values)}
        elif isinstance(value, (list, tuple, set)):
            logger.debug("Resolving collection: %s", value)
            if isinstance(value, list):
                return await asyncio.gather(
                    *(self._resolve_nested(item) for item in value)
                )
            elif isinstance(value, tuple):
                return tuple(
                    await asyncio.gather(
                        *(self._resolve_nested(item) for item in value)
                    )
                )
            elif isinstance(value, set):
                return set(
                    await asyncio.gather(
                        *(self._resolve_nested(item) for item in value)
                    )
                )
        else:
            return value

    async def evaluate(self) -> Any:
        if not self._evaluated:
            resolved_args = await self._resolve_nested(self.args)
            resolved_kwargs = await self._resolve_nested(self.kwargs)
            logger.debug("Resolved args: %s", resolved_args)
            logger.debug("Resolved kwargs: %s", resolved_kwargs)
            result = await await_if_needed(self.func(*resolved_args, **resolved_kwargs))
            logger.debug(f"Ran {self.name} func with a result {result}")

            # Resolve if the entire result is lazy
            while hasattr(result, "execute_async") and callable(result.execute_async):
                result = await result.execute_async()
            # Then recursively resolve nested lazy objects
            result = await self._resolve_nested(result)

            self._result = result
            # Potentially handle the case of result being lazy here. Not sure if it's necessary
            self._evaluated = True
        return self._result

    async def execute_async(self) -> Any:
        return await self.evaluate()

    def execute(self) -> Any:
        return run_async(self.execute_async())

    @model_serializer
    def serialize_model(self) -> dict:
        """Only serialize the name, not the problematic object"""
        return {"name": self.name}
