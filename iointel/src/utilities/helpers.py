import asyncio
import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, model_serializer

import logging
import os
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Fallback to "DEBUG" if not set
level_name = os.environ.get("LOGGING_LEVEL", "INFO")
level_name = level_name.upper()
# Safely get the numeric logging level, default to DEBUG if invalid
numeric_level = getattr(logging, level_name, logging.INFO)
logger.setLevel(numeric_level)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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
        self._evaluated = False
        self._result = None

    def _resolve_nested(self, value: Any) -> Any:
        logger.debug("Resolving: %s", value)
        if hasattr(value, "execute") and callable(value.execute):
            logger.debug("Resolving lazy object: %s", value)
            resolved = self._resolve_nested(value.execute())
            logger.debug("Resolved lazy object to: %s", resolved)
            return resolved
        elif isinstance(value, dict):
            logger.debug("Resolving dict: %s", value)
            return {k: self._resolve_nested(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            logger.debug("Resolving collection: %s", value)
            t = type(value)
            if isinstance(value, list):
                return [self._resolve_nested(item) for item in value]
            elif isinstance(value, tuple):
                return tuple(self._resolve_nested(item) for item in value)
            elif isinstance(value, set):
                return {self._resolve_nested(item) for item in value}
        else:
            return value

    def evaluate(self) -> Any:
        if not self._evaluated:
            resolved_args = self._resolve_nested(self.args)
            resolved_kwargs = self._resolve_nested(self.kwargs)
            logger.debug("Resolved args: %s", resolved_args)
            logger.debug("Resolved args: %s", resolved_args)
            result = self.func(*resolved_args, **resolved_kwargs)
            if inspect.isawaitable(result):
                try:
                    result = asyncio.run(result)
                except RuntimeError:
                    raise RuntimeError(
                        "Lazy function returned an awaitable; please await it externally."
                    )
            self._result = result
            self._evaluated = True
        return self._result

    def execute(self) -> Any:
        result = self.evaluate()
        # Resolve if the entire result is lazy
        while hasattr(result, "execute") and callable(result.execute):
            result = result.execute()
        # Then recursively resolve nested lazy objects
        return self._resolve_nested(result)

    @model_serializer
    def ser_model(self) -> dict:
        """Only serialize the name, not the problematic object"""
        return {"name": self.name}
