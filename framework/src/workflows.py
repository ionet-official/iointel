# A global or module-level registry of custom workflows
CUSTOM_WORKFLOW_REGISTRY = {}

def register_custom_workflow(name: str):
    def decorator(func):
        CUSTOM_WORKFLOW_REGISTRY[name] = func
        return func
    return decorator