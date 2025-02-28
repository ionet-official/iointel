from typing import Dict, Callable

# A global registry mapping task types to executor functions.
TASK_EXECUTOR_REGISTRY: Dict[str, Callable] = {}

# A global registry mapping chainable method names to functions.
CHAINABLE_METHODS: Dict[str, Callable] = {}

# A global or module-level registry of custom workflows
CUSTOM_WORKFLOW_REGISTRY: Dict[str, Callable] = {}

# A global or module-level registry of custom tools
TOOLS_REGISTRY: Dict[str, Callable] = {}
