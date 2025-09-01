from typing import Any, Dict, Callable, TYPE_CHECKING


if TYPE_CHECKING:
    from pydantic import BaseModel
    from ..agent_methods.data_models.datamodels import Tool

# A global registry mapping task types to executor functions.
TASK_EXECUTOR_REGISTRY: Dict[str, Callable] = {}

# A global registry mapping chainable method names to functions.
CHAINABLE_METHODS: Dict[str, Callable] = {}

# A global or module-level registry of custom workflows
CUSTOM_WORKFLOW_REGISTRY: Dict[str, Callable] = {}

# A global or module-level registry of custom tools
TOOLS_REGISTRY: "Dict[str, Tool]" = {}

# A global registry of classes which instance methods are registered as tools
TOOL_SELF_REGISTRY: "Dict[str, type[BaseModel]]" = {}

# The registry for initialising default state for stateful tools.
# Maps tool name -> tool arg -> arg value
STATEFUL_TOOL_DEFAULTS: Dict[str, Dict[str, Any]] = {}
