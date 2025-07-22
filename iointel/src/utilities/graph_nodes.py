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

    @classmethod
    def get_id(cls) -> str:
        return "WorkflowState"


@dataclass
class TaskNode(BaseNode[WorkflowState]):
    task: dict
    default_text: str
    default_agents: list
    conversation_id: str
    unique_id: str = field(init=False)
    next_task: Optional["TaskNode"] = None

    def __post_init__(self):
        self.unique_id = (
            self.task.get("task_id") or self.task.get("name") or str(uuid.uuid4())
        )

        self.conversation_id = self.conversation_id

    @classmethod
    def get_node_def(cls, local_ns: Optional[Dict[str, Any]] = None):
        next_edges: dict[str, Any] = {}
        next_cls = getattr(cls, "next_task", None)
        if next_cls is not None:
            next_edges[next_cls.get_id()] = None
        return NodeDef(
            node=cls,
            node_id=cls.get_id(),
            note="",
            next_node_edges=next_edges,
            end_edge=None,
            returns_base_node=True,
        )

    async def run(
        self, context: GraphRunContext[WorkflowState]
    ) -> "TaskNode" | End[WorkflowState]:
        from ..workflow import (
            Workflow,
            _get_task_key,
        )  # import must happen here, or circular issue occurs
        from .data_flow_resolver import data_flow_resolver

        wf = Workflow()
        state = context.state

        if not state.conversation_id:
            state.conversation_id = self.conversation_id or str(uuid.uuid4())
            self.conversation_id = state.conversation_id

        self.task["conversation_id"] = state.conversation_id
        
        # Also add conversation_id to task_metadata if it exists
        if "task_metadata" in self.task and isinstance(self.task["task_metadata"], dict):
            self.task["task_metadata"]["conversation_id"] = state.conversation_id

        task_key = _get_task_key(self.task)

        # 🔧 NEW: Resolve variable references in task metadata before execution
        resolved_task = self.task.copy()
        if state.results and "task_metadata" in resolved_task:
            task_metadata = resolved_task["task_metadata"]
            if isinstance(task_metadata, dict):
                try:
                    print(f"   📊 Resolving variables for task '{task_key}'")
                    print(f"   📊 Available results: {list(state.results.keys())}")
                    
                    # Resolve variables in config
                    if "config" in task_metadata:
                        print(f"   📊 Original config: {task_metadata['config']}")
                        resolved_config = data_flow_resolver.resolve_config(
                            task_metadata["config"], state.results
                        )
                        resolved_task["task_metadata"] = task_metadata.copy()
                        resolved_task["task_metadata"]["config"] = resolved_config
                        print(f"   ✅ Resolved config: {resolved_config}")
                    
                    # Resolve variables in agent_instructions for agent tasks
                    if "agent_instructions" in task_metadata and isinstance(task_metadata["agent_instructions"], str):
                        print(f"   📊 Original agent instructions: {task_metadata['agent_instructions']}")
                        resolved_instructions = data_flow_resolver._resolve_value(
                            task_metadata["agent_instructions"], state.results
                        )
                        if "task_metadata" not in resolved_task:
                            resolved_task["task_metadata"] = task_metadata.copy()
                        resolved_task["task_metadata"]["agent_instructions"] = resolved_instructions
                        print(f"   ✅ Resolved agent instructions: {resolved_instructions}")
                        
                except Exception as e:
                    print(f"   ⚠️  Variable resolution failed for '{task_key}': {e}")
                    # Continue with original task if resolution fails

        # Add available results to the task for agent context
        if resolved_task.get("task_metadata") and state.results:
            if "available_results" not in resolved_task["task_metadata"]:
                resolved_task["task_metadata"]["available_results"] = state.results.copy()
        
        # Pass agent_result_format if it was set on this node
        agent_result_format = getattr(self, '_agent_result_format', 'full')
        # print(f"🔧 graph_nodes: using agent_result_format = {agent_result_format}")
        
        result = await wf.run_task(
            resolved_task, self.default_text, self.default_agents, self.conversation_id,
            agent_result_format=agent_result_format
        )

        # Store the core result value for data flow
        if isinstance(result, dict):
            # Extract the main value from different result formats
            if "result" in result:
                # Standard format: {"result": value, ...}
                core_value = result["result"]
            elif "user_input" in result:
                # User input format: {"user_input": value, ...}
                core_value = result["user_input"]
            elif "data" in result:
                # Legacy format: {"data": value, ...}
                core_value = result["data"]
            else:
                # For complex results, store the whole dict
                core_value = result
        else:
            # Simple value, store directly
            core_value = result
            
        state.results[task_key] = core_value
        print(f"   💾 Stored '{task_key}' = {core_value} (type: {type(core_value)})")
        return self.next_task if self.next_task else End(state)


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
