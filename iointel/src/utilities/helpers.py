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
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = make_logger(__name__)


class LazyCaller(BaseModel):
    func: Callable
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    _evaluated: bool = False
    _result: Any = None
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(func=func, args=args, kwargs=kwargs, name=func.__name__)
        self._async_caller = AsyncLazyCaller(self.func, self.args, self.kwargs)

    def execute(self) -> Any:
        if self._evaluated:
            return self._result
        self._result = run_async(self._async_caller.execute())
        self._evaluated = True
        return self._result

    @model_serializer
    def set_model(self) -> dict:
        """Only serialize the name, not the problematic object"""
        return {"name": self.name}


class AsyncLazyCaller(BaseModel):
    func: Callable
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    _evaluated: bool = False
    _result: Any = None
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(func=func, args=args, kwargs=kwargs, name=func.__name__)
        self._evaluated = False
        self._result = None

    async def _resolve_nested(self, value: Any) -> Any:
        logger.debug("Resolving: %s", value)
        if hasattr(value, "execute") and callable(value.execute):
            logger.debug("Resolving lazy object: %s", value)
            result = await await_if_needed(value.execute())
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
            resolved_args = self._resolve_nested(self.args)
            resolved_kwargs = self._resolve_nested(self.kwargs)
            logger.debug("Resolved args: %s", resolved_args)
            logger.debug("Resolved kwargs: %s", resolved_kwargs)
            self._result = await await_if_needed(
                self.func(*resolved_args, **resolved_kwargs)
            )
            # Potentially handle the case of result being lazy here. Not sure if it's necessary
            self._evaluated = True
        return self._result

    async def execute(self) -> Any:
        return await self.evaluate()

    @model_serializer
    def set_model(self) -> dict:
        """Only serialize the name, not the problematic object"""
        return {"name": self.name}
