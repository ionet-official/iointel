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

    @register_tool(name="read_context_tree")
    def read(self, node_id: str, as_content: bool = True) -> Optional[str | ContextNode]:
        """Return the content of a node. If as_content is True, return the content of the node. Otherwise, return the node itself.
        
        Args:
            node_id: The unique identifier of the node to read
            as_content: If True, return only the content string; if False, return the full node object
        
        Returns:
            The node content as string (if as_content=True) or the ContextNode object (if as_content=False)
        """
        if as_content:
            return self._index[node_id].content
        else:
            return self._index[node_id]

    @register_tool(name="create_context_tree")
    def create(
        self,
        parent_id: str,
        title: str,
        content: Optional[str],
        deletable: bool = False,
    ) -> ContextNode:
        """Create a new node under parent_id. If you unsure about the parent_id, use `summary` to get a list of nodes and their ids.
        
        Args:
            parent_id: The ID of the parent node to create the new node under
            title: A concise title for the new node (max 3-4 words recommended)
            content: The initial content for the new node (can be None)
            deletable: Whether this node can be deleted later (default: False)
        
        Returns:
            The newly created ContextNode
        """
        node = ContextNode.create(
            title=title, content=content, id_length=self.id_length, deletable=deletable
        )
        self._index[parent_id].children.append(node)
        self._index[node.id] = node
        return node

    @register_tool(name="append_context_tree")
    def append(self, node_id: str, content: str) -> ContextNode:
        """Append content to an existing node. This lets you grow the node's content iteratively.
        
        Args:
            node_id: The unique identifier of the node to append content to
            content: The content to append to the existing node content
        
        Returns:
            The updated ContextNode with appended content
        """
        node = self._index[node_id]
        node.content = (node.content or "") + "\n" + content
        return node

    @register_tool(name="update_context_tree")
    def update(self, node_id: str, content: str) -> ContextNode:
        """Overwrite the content of an existing node. Useful for updating checklists, and running todo lists. This will replace the entire node with the new content.
        
        Args:
            node_id: The unique identifier of the node to update
            content: The new content that will completely replace the existing content
        
        Returns:
            The updated ContextNode with new content
        """
        node = self._index[node_id]
        node.content = content
        return node

    @register_tool(name="delete_context_tree")
    def delete(self, node_id: str) -> ContextNode | str:
        """Delete a node if deletable. Returns the deleted node or a message if not deletable.
        
        Args:
            node_id: The unique identifier of the node to delete
        
        Returns:
            The deleted ContextNode if successful, or an error message string if not deletable
        """
        node = self._index[node_id]
        if not node.deletable:
            return f"Node {node_id} is not deletable"
        del self._index[node_id]
        return node

    @register_tool(name="summary_context_tree")
    def summary(
        self,
        start_node_id: str = "root",
        max_depth: Optional[int] = None,
        show_content: bool = False,
        return_type: Literal["str", "dict"] = "str",
    ) -> Union[str, dict]:
        """Return a pretty-printed, indented summary of the tree or subtree starting from start_node_id.
        
        Args:
            start_node_id: The node ID to start the summary from (default: "root")
            max_depth: Maximum depth to traverse (None for unlimited)
            show_content: Whether to include node content in the summary
            return_type: Format of return value - 'str' for pretty string, 'dict' for nested dict
        
        Returns:
            A formatted tree summary as string or dict, showing node structure and deletability
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

    @register_tool(name="load_context_tree")
    def load_tree(self, file_path: str) -> str:
        """Load a tree from a file.
        
        Args:
            file_path: The path to the JSON file containing the serialized context tree
        
        Returns:
            A confirmation message indicating successful loading
        """
        with open(file_path) as f:
            self.root = ContextNode.model_validate(json.load(f))
        self._rebuild_index()
        return f"Loaded tree from {file_path}"

    @register_tool(name="save_context_tree")        
    def save_tree(self, file_path: str) -> str:
        """Save the tree to a file.
        
        Args:
            file_path: The path where the context tree should be saved as JSON
        
        Returns:
            A confirmation message indicating successful saving
        """
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

tree = ContextTree()

CONTEXT_AGENT_INSTRUCTIONS = """
You are a context-librarian. Given a query, output a directing action, one of:
- create: Add a new node under a parent (use a concise max 3-4 word title). Make sure to think if node is ephemeral or permanent carefully. ToDo lists etc are likely ephemeral.
- read: Return the content of a node.
- append: Add content to an existing node.
- update: Overwrite the entire content of an existing node.
- delete: Remove a node if it is deletable.
- summary: Provide an indented summary of the tree (use show_content=True to include node contents).
- load_tree: Load a tree from a file.
- save_tree: Save the tree to a file.
with appropriate context. 

You enjoy writing and can format content in Markdown. Preserve all important details, and when appropriate,
use checkboxes for TODO lists.
"""


def get_tree_agent(model_name: str, api_key: str, base_url: str, conversation_id: str, id_length: int = 7, memory: Optional[AsyncMemory] = None) -> Agent:
    tree = ContextTree(id_length=id_length)
    agent = Agent(
    name="Context Tree Agent",
    instructions=CONTEXT_AGENT_INSTRUCTIONS,
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    conversation_id=conversation_id,
    memory=memory,
    tools=[
        tree.read,
        tree.create,
        tree.append,
        tree.update,
        tree.delete,
        tree.summary,
        tree.load_tree,
        tree.save_tree,
    ],
    show_tool_calls=True,
    tool_pil_layout="horizontal",
    debug=False,
)
    return agent
