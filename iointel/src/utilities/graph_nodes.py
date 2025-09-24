from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.nodes import NodeDef
from typing import Any, Dict, Optional
import uuid
from dataclasses import dataclass, field, make_dataclass


@dataclass
class WorkflowState:
    initial_text: str = ""
    conversation_id: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    user_inputs: Dict[str, Any] = field(default_factory=dict)  # Runtime user inputs
    execution_id: Optional[str] = None  # Unique execution ID for tracking
    execution_summary: Optional[Any] = None  # Execution summary from feedback collector

    @classmethod
    def get_id(cls) -> str:
        return "WorkflowState"


# @dataclass
class TaskNode(BaseNode[WorkflowState]):
    """Legacy TaskNode for old workflow system - kept for backward compatibility."""
    task: dict
    default_text: str
    default_agents: list
    conversation_id: str
    unique_id: str = field(init=False)
    next_task: Optional["TaskNode"] = None

#     def __post_init__(self):
#         self.unique_id = (
#             self.task.get("task_id") or self.task.get("name") or str(uuid.uuid4())
#         )

#         self.conversation_id = self.conversation_id

#     @classmethod
#     def get_node_def(cls, local_ns: Optional[Dict[str, Any]] = None) -> NodeDef:
#         next_edges: dict[str, Any] = {}
#         next_cls = getattr(cls, "next_task", None)
#         if next_cls is not None:
#             next_edges[next_cls.get_id()] = None
#         return NodeDef(
#             node=cls,
#             node_id=cls.get_id(),
#             note="",
#             next_node_edges=next_edges,
#             end_edge=None,
#             returns_base_node=True,
#         )

    async def run(
        self, context: "GraphRunContext[WorkflowState]"
    ) -> "TaskNode | End[WorkflowState]":
        """
        Legacy TaskNode execution - kept for backward compatibility.
        
        For new code, use the DAG executor with pure Pydantic models instead.
        """
        # Minimal implementation to prevent import errors
        # Full implementation would be restored here if needed for legacy code
        raise NotImplementedError("Legacy TaskNode execution - use DAG executor with pure Pydantic models for new code")


def make_task_node(
    task: dict, default_text: str, default_agents: list, conv_id: str
) -> type[BaseNode]:
    """
    Return a brand‑new subclass of TaskNode whose `get_id()` is unique.
    The task parameters are stored on the *class* so the graph can later
    instantiate it.
    """
    uid = task.get("task_id") or task.get("name") or str(uuid.uuid4())
    cls_name = f"{task.get('name', 'task')}_{uid}".replace("-", "_")

    # Every field you’d pass to TaskNode’s __init__ becomes a *class*
    # attribute; pydantic‑graph will pass them to the generated __init__.
    # ``make_dataclass`` requires a *fields* argument; since all the
    # data‑carrying fields are already inherited from the base ``TaskNode``
    # we can simply pass an **empty list**.  (Leaving it out raises the
    # ``TypeError: make_dataclass() missing 1 required positional argument: 'fields'``)

    NewNode = make_dataclass(
        cls_name,
        [],  # <‑‑ empty ``fields`` list satisfies the API
        bases=(TaskNode,),
        namespace={
            # class‑level constants → become default values for the subclass
            "task": task,
            "default_text": default_text,
            "default_agents": default_agents,
            "conversation_id": conv_id,
            "_node_id": cls_name,  # used by the custom ``get_id`` below
            # Provide a per‑class ``get_id`` so every generated subclass
            # advertises a unique node‑ID to pydantic‑graph
            "get_id": classmethod(lambda cls: cls._node_id),
        },
    )
    return NewNode
