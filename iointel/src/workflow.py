# Legacy workflow system - kept for backward compatibility
# New code should use the DAG executor with pure Pydantic models

from .utilities.graph_nodes import WorkflowState, TaskNode, make_task_node
from pydantic_graph import Graph, End, GraphRunContext

def _get_task_key(task: dict) -> str:
    """
    Generate a unique key for a task based on its content.
    """
    return (
        task.get("task_id")
        or task.get("name")
        or task.get("type")
        or "task"
    )


class Workflow:
    """
    Legacy workflow system - kept for backward compatibility.
    
    For new projects, use the DAG executor with pure Pydantic models instead.
    This class is maintained for existing code that depends on it.
    """
    
    def __init__(self, *args, **kwargs):
        # Keep the old workflow system working for backward compatibility
        pass
    
    # Minimal implementation to prevent import errors
    # Full implementation would be restored here if needed
