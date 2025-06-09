from typing import Literal, Optional, List, Union
from pydantic import BaseModel, PrivateAttr
import json
import random
import string

from iointel.src.memory import AsyncMemory
from iointel.src.agents import Agent
from iointel import register_tool


class ContextNode(BaseModel):
    id: str
    title: str
    content: Optional[str] = None
    children: List["ContextNode"] = []
    deletable: bool = False

    @classmethod
    def create(
        cls,
        title: str,
        content: Optional[str] = None,
        id_length: int = 7,
        deletable: bool = False,
    ) -> "ContextNode":
        def short_id(length=id_length):
            return "".join(
                random.choices(string.ascii_lowercase + string.digits, k=length)
            )

        return cls(
            id=short_id(),
            title=title,
            content=content,
            children=[],
            deletable=deletable,
        )


ContextNode.update_forward_refs()


class ContextCommand(BaseModel):
    action: Literal[
        "read",
        "create",
        "append",
        "update",
        "delete",
        "summary",
        "load_tree",
        "save_tree",
    ]
    node_id: Optional[str] = None  # target node
    title: Optional[str] = None  # for create
    content: Optional[str] = None  # for write or create


class ContextTree(BaseModel):
    root: ContextNode = None
    id_length: int = 7
    _index: dict = PrivateAttr(default_factory=dict)

    def _rebuild_index(self):
        self._index = {}

        def index_nodes(node):
            self._index[node.id] = node
            for child in node.children:
                index_nodes(child)

        index_nodes(self.root)

    def model_post_init(self, __context) -> None:
        if self.root is None:
            self.root = ContextNode(
                id="root", title="A Self-Writing Wiki", deletable=False
            )
        self._rebuild_index()

    @register_tool
    def read(self, node_id: str, as_content: bool = False) -> Optional[str]:
        """
        Return the content of a node. If as_content is True, return the content of the node. Otherwise, return the node itself.
        """
        if as_content:
            return self._index[node_id].content
        else:
            return self._index[node_id]

    @register_tool
    def create(
        self,
        parent_id: str,
        title: str,
        content: Optional[str],
        deletable: bool = False,
    ) -> ContextNode:
        """Create a new node under parent_id. If you unsure about the parent_id, use `summary` to get a list of nodes and their ids."""
        node = ContextNode.create(
            title=title, content=content, id_length=self.id_length, deletable=deletable
        )
        self._index[parent_id].children.append(node)
        self._index[node.id] = node
        return node

    @register_tool
    def append(self, node_id: str, content: str) -> ContextNode:
        """Append content to an existing node. This lets you grow the node's content iteratively."""
        node = self._index[node_id]
        node.content = (node.content or "") + "\n" + content
        return node

    @register_tool
    def update(self, node_id: str, content: str) -> ContextNode:
        """Overwrite the content of an existing node. Useful for updating checklists, and running todo lists. This will replace the entire node with the new content."""
        node = self._index[node_id]
        node.content = content
        return node

    @register_tool
    def delete(self, node_id: str) -> ContextNode:
        """Delete a node if deletable. Returns the deleted node."""
        node = self._index[node_id]
        if not node.deletable:
            raise ValueError(f"Node {node_id} is not deletable")
        del self._index[node_id]
        return node

    @register_tool
    def summary(
        self,
        start_node_id: str = "root",
        max_depth: int = None,
        show_content: bool = False,
        return_type: Literal["str", "dict"] = "str",
    ) -> Union[str, dict]:
        """
        Return a pretty-printed, indented summary of the tree or subtree starting from start_node_id.
        - max_depth: limit depth (None for unlimited)
        - show_content: include content in summary
        - return_type: 'str' for pretty string, 'dict' for nested dict
        Each node includes whether it is deletable.
        """
        node = self._index.get(start_node_id, self.root)

        def walk_str(node, indent=0, depth=0):
            if max_depth is not None and depth > max_depth:
                return []
            prefix = "  " * indent
            line = f"{prefix}-id: {node.id}, title: {node.title}"
            if node.deletable:
                line += " (deletable)"
            if show_content and node.content:
                line += f"\n{prefix} ==content==:\n{node.content}"
            lines = [line]
            for child in node.children:
                lines.extend(walk_str(child, indent + 1, depth + 1))
            return lines

        def walk_dict(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return None
            children = [
                c
                for c in (walk_dict(child, depth + 1) for child in node.children)
                if c is not None
            ]
            return {
                "id": node.id,
                "title": node.title,
                "content": node.content if show_content else None,
                "children": children,
                "deletable": node.deletable,
            }

        if return_type == "str":
            return "\n".join(walk_str(node))
        elif return_type == "dict":
            d = walk_dict(node)
            return d if d is not None else {}
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    @register_tool
    def load_tree(self, file_path: str) -> str:
        """Load a tree from a file."""
        with open(file_path, "r") as f:
            self.root = ContextNode.model_validate(json.load(f))
        self._rebuild_index()
        return f"Loaded tree from {file_path}"

    @register_tool
    def save_tree(self, file_path: str) -> str:
        """Save the tree to a file."""
        with open(file_path, "w") as f:
            json.dump(self.serialize(), f, indent=2)
        return f"Saved tree to {file_path}"

    def serialize(self) -> dict:
        """Serialize the tree to a dict."""
        return self.root.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ContextTree":
        """Deserialize a tree from a dict."""
        tree = cls()
        tree.root = ContextNode.model_validate(data)
        tree._rebuild_index()
        return tree


CONTEXT_AGENT_INSTRUCTIONS = """
You are a context-librarian. Given a query, output a ContextCommand JSON directing action, one of:
- create: Add a new node under a parent (use a concise max 3-4 word title).
- read: Return the content of a node.
- append: Add content to an existing node.
- update: Overwrite the entire content of an existing node.
- delete: Remove a node if it is deletable.
- summary: Provide an indented summary of the tree (use show_content=True to include node contents).

You enjoy writing and can format content in Markdown. Preserve all important details, and when appropriate,
use checkboxes for TODO lists.
"""


class ContextTreeAgent:
    """
    A context tree agent that can be used to create, read, update, and delete nodes in a context tree. Convience wrapper around the ContextTree + Agent class.
    """

    def __init__(
        self,
        id_length: int = 7,
        model_name: str = "gpt-4o",
        api_key: str = None,
        base_url: str = None,
        memory: Optional[AsyncMemory] = None,
        save_each_turn: bool = False,
    ) -> None:
        """
        Initialize a context tree agent.
        - id_length: the length of the node ids
        - model_name: the model to use
        - api_key: the api key to use
        - base_url: the base url to use
        - memory: the memory to use
        - save_each_turn: whether to save the tree after each turn
        """
        self.tree = ContextTree(id_length=id_length)
        self.agent = Agent(
            name="Context Tree Agent",
            instructions=CONTEXT_AGENT_INSTRUCTIONS,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            memory=memory,
            tools=[
                self.tree.read,
                self.tree.create,
                self.tree.append,
                self.tree.update,
                self.tree.delete,
                self.tree.summary,
                self.tree.load_tree,
                self.tree.save_tree,
            ],
            show_tool_calls=True,
            tool_pil_layout="horizontal",
            debug=False,
        )
        self.save_each_turn = save_each_turn

    def run(self, query: str, conversation_id: str = None, pretty: bool = True) -> str:
        """
        query the context tree.
        """
        tree_summary = self.tree.summary(return_type="str")
        result = self.agent.run(
            query, context=tree_summary, conversation_id=conversation_id, pretty=pretty
        )

        if self.save_each_turn:
            self.tree.save_tree(f"_context_tree_{conversation_id}.json")

        return result
