"""
WorkflowSpec: Clean, discriminated union-based workflow specification.

This module defines the workflow specification with proper separation of concerns:
- data_source nodes: User input or static prompts (no tools)
- agent nodes: Tool-using nodes with optional SLA
- decision nodes: Agents with enforced routing_gate routing
- workflow_call nodes: Sub-workflow invocation

Routing information lives on edges, not nodes.
"""
from __future__ import annotations


from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from iointel.src.agent_methods.data_models.data_source_registry import ValidDataSourceName

# -----------------------------
# Core constants & type aliases
# -----------------------------

# Data sources are NOT tools - they're input mechanisms
_DATA_SOURCE_NAMES: set[str] = {"user_input", "prompt_tool"}

# Centralized routing tools definition (simplified to routing_gate only)
ROUTING_TOOLS = ['routing_gate']


# -----------------------------
# Test alignment (unchanged)
# -----------------------------

class TestResult(BaseModel):
    """Result of running a test against a workflow."""
    test_id: str
    test_name: str
    passed: bool
    executed_at: datetime
    execution_details: dict | None = None
    error_message: str | None = None


# class TestAlignment(BaseModel):
#     """Test alignment metadata for workflows."""
#     test_ids: set[str] = Field(default_factory=set, description="Test IDs that validate this workflow")
#     test_results: list[TestResult] = Field(default_factory=list, description="Historical test results")
#     last_validated: datetime | None = None
#     validation_status: Literal["untested", "passing", "failing", "mixed"] = "untested"
#     production_ready: bool = Field(default=False, description="True if all critical tests pass")
    
#     def add_test_result(self, result: TestResult):
#         """Add a new test result and update validation status."""
#         self.test_results.append(result)
#         self.test_ids.add(result.test_id)
#         self.last_validated = result.executed_at
#         self._update_validation_status()
    
#     def _update_validation_status(self):
#         """Update validation status based on latest test results."""
#         if not self.test_results:
#             self.validation_status = "untested"
#             self.production_ready = False
#             return
        
#         # Get latest result for each test
#         latest_results = {}
#         for result in self.test_results:
#             if result.test_id not in latest_results or result.executed_at > latest_results[result.test_id].executed_at:
#                 latest_results[result.test_id] = result
        
#         passed = sum(1 for r in latest_results.values() if r.passed)
#         total = len(latest_results)
        
#         if passed == total:
#             self.validation_status = "passing"
#             self.production_ready = True
#         elif passed == 0:
#             self.validation_status = "failing"
#             self.production_ready = False
#         else:
#             self.validation_status = "mixed"
#             self.production_ready = False


# -----------------------------
# SLA Requirements
# -----------------------------

class SLARequirements(BaseModel):
    """
    Service-level constraints for an AGENT or DECISION node.
    
    This object exists ONLY to constrain behavior of tool-using nodes.
    If you specify required tools or final tool requirements, set `enforce_usage=True`.
    """
    tool_usage_required: bool = Field(
        default=True,
        description="If true, the node MUST call at least one tool during execution."
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Exact tool names that MUST be called at least once."
    )
    final_tool_must_be: str | None = Field(
        default=None,
        description="Exact tool name that MUST be the LAST call (e.g., 'routing_gate' for decision routing)."
    )
    min_tool_calls: int = Field(
        default=1,
        ge=0,
        description="Minimum number of tool calls required to satisfy the SLA."
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Maximum retry attempts for the node (0–3)."
    )
    timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=300,
        description="Maximum time allowed for the node to complete (seconds)."
    )
    enforce_usage: bool = Field(
        default=True,
        description="If true, the executor will enforce all SLA rules above."
    )


# -----------------------------
# Data source config
# -----------------------------

class DataSourceConfig(BaseModel):
    """
    REQUIRED configuration shape for ANY data_source node.
    
    The agent MUST provide BOTH keys exactly:
    - message: prompt shown to the user
    - default_value: fallback value used if user provides nothing
    """
    message: str = Field(description="Prompt message displayed to the user.")
    default_value: str = Field(description="Fallback value if the user supplies nothing.")


# -----------------------------
# Node payloads (data) by type
# -----------------------------

class DataSourceData(BaseModel):
    """
    Payload for data_source nodes.
    
    Separation of concerns:
    - `source_name` MUST be one of ('user_input', 'prompt_tool')
    - `config` MUST match DataSourceConfig (message + default_value)
    """
    source_name: ValidDataSourceName = Field(
        description="Name of the data source to invoke. ONLY 'user_input' or 'prompt_tool' are valid."
    )
    config: DataSourceConfig = Field(
        description="Configuration for the data source. MUST include 'message' and 'default_value'."
    )


class AgentConfig(BaseModel):
    """
    Payload for standard agent nodes.
    
    Separation of concerns:
    - DO NOT include data sources in `tools`
    - Use SLA when you need to force tool usage patterns
    """
    agent_instructions: str = Field(
        description="Clear, actionable instructions for the agent. Reference upstream labels as needed."
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Exact tool names the agent MAY call. NEVER include data sources here."
    )
    model: str | None = Field(
        default="gpt-4o",
        description="Model to use for the agent (if applicable)."
    )
    config: dict = Field(
        default_factory=dict,
        description="Optional agent parameters (hyperparameters, constants, etc.)."
    )
    sla: SLARequirements | None = Field(
        default=None,
        description="Optional SLA to enforce tool usage/ordering/timeouts for this agent."
    )


class DecisionConfig(AgentConfig):
    """
    Payload for decision nodes.
    
    REQUIRED invariants (routing-specific):
    - tools MUST include 'routing_gate'
    - sla MUST exist
    - sla.enforce_usage MUST be true
    - sla.required_tools MUST include 'routing_gate'
    - sla.final_tool_must_be MUST equal 'routing_gate'
    """
    @model_validator(mode="after")
    def _decision_requirements(self) -> DecisionConfig:
        tools_set = set(self.tools or [])
        
        if "routing_gate" not in tools_set:
            raise ValueError("DecisionConfig.tools MUST include 'routing_gate'.")
        
        if self.sla is None:
            raise ValueError("DecisionConfig.sla is REQUIRED.")
        if not self.sla.enforce_usage:
            raise ValueError("DecisionConfig.sla.enforce_usage MUST be true.")
        if "routing_gate" not in set(self.sla.required_tools or []):
            raise ValueError("DecisionConfig.sla.required_tools MUST include 'routing_gate'.")
        if self.sla.final_tool_must_be != "routing_gate":
            raise ValueError("DecisionConfig.sla.final_tool_must_be MUST be 'routing_gate'.")
        return self


# -----------------------------
# Node specs (discriminated) for LLM
# -----------------------------

class DataSourceNodeLLM(BaseModel):
    """
    Node of type 'data_source' (LLM version without ID).
    
    Purpose:
    - Collect user input or inject static prompt/context.
    - NEVER perform API calls here; those belong to agents.
    """
    type: Literal["data_source"] = Field("data_source", description="Discriminator: 'data_source'.")
    label: str = Field(description="Human-readable node label. Used for referencing in instructions.")
    data: DataSourceData = Field(description="Configuration for the data source.")


class AgentNodeLLM(BaseModel):
    """
    Node of type 'agent' (LLM version without ID).
    
    Purpose:
    - Use tools and reasoning to transform inputs into outputs.
    - May include an SLA if tool usage must be enforced.
    """
    type: Literal["agent"] = Field("agent", description="Discriminator: 'agent'.")
    label: str = Field(description="Human-readable node label.")
    data: AgentConfig = Field(description="Agent configuration.")


class DecisionNodeLLM(BaseModel):
    """
    Node of type 'decision' (LLM version without ID).
    
    Purpose:
    - Make a routing decision using 'routing_gate'.
    - MUST include SLA that enforces 'routing_gate' as the final tool.
    Routing is expressed ONLY on the outgoing edges via route_index (0..N).
    """
    type: Literal["decision"] = Field("decision", description="Discriminator: 'decision'.")
    label: str = Field(description="Human-readable node label.")
    data: DecisionConfig = Field(description="Decision configuration with enforced gate + SLA using routing_gate.")


class WorkflowCallNodeLLM(BaseModel):
    """
    Node of type 'workflow_call' (LLM version without ID).
    
    Purpose:
    - Invoke a named sub-workflow by ID. Keep this minimal.
    """
    type: Literal["workflow_call"] = Field("workflow_call", description="Discriminator: 'workflow_call'.")
    label: str = Field(description="Human-readable node label.")
    data: dict[str, str] = Field(
        description="MUST include {'workflow_id': '<id>'}. Additional static config is allowed."
    )


NodeSpecLLM = Annotated[
    Union[DataSourceNodeLLM, AgentNodeLLM, DecisionNodeLLM, WorkflowCallNodeLLM],
    Field(discriminator="type", description="Union of supported node types, discriminated by 'type'.")
]


# -----------------------------
# Node specs with IDs (for execution)
# -----------------------------

class DataSourceNode(BaseModel):
    """Node of type 'data_source' with ID."""
    id: str
    type: Literal["data_source"] = "data_source"
    label: str
    data: DataSourceData
    position: dict[str, float] | None = None
    runtime: dict = Field(default_factory=dict)
    sla: SLARequirements | None = None


class AgentNode(BaseModel):
    """Node of type 'agent' with ID."""
    id: str
    type: Literal["agent"] = "agent"
    label: str
    data: AgentConfig
    position: dict[str, float] | None = None
    runtime: dict = Field(default_factory=dict)
    sla: SLARequirements | None = None


class DecisionNode(BaseModel):
    """Node of type 'decision' with ID."""
    id: str
    type: Literal["decision"] = "decision"
    label: str
    data: DecisionConfig
    position: dict[str, float] | None = None
    runtime: dict = Field(default_factory=dict)
    sla: SLARequirements | None = None


class WorkflowCallNode(BaseModel):
    """Node of type 'workflow_call' with ID."""
    id: str
    type: Literal["workflow_call"] = "workflow_call"
    label: str
    data: dict[str, str]
    position: dict[str, float] | None = None
    runtime: dict = Field(default_factory=dict)
    sla: SLARequirements | None = None


NodeSpec = Annotated[
    Union[DataSourceNode, AgentNode, DecisionNode, WorkflowCallNode],
    Field(discriminator="type", description="Union of supported node types with IDs."),
]


# -----------------------------
# Edge specs
# -----------------------------

class EdgeData(BaseModel):
    """
    Data structure for React Flow edge configuration.
    
    Routing information for decision nodes:
    - route_index: Required for edges from decision nodes (0, 1, 2...)
    - route_label: Optional human-readable label
    """
    route_index: int | None = Field(
        None, 
        ge=0,
        description="REQUIRED only when source is a DECISION node. Index of the routed branch (0..N)."
    )
    route_label: str | None = Field(
        None,
        description="OPTIONAL human-friendly name for the branch (e.g., 'buy', 'sell')."
    )
    # Legacy support - will be deprecated
    condition: str | None = Field(None, description="Legacy condition string - use route_index instead")


class EdgeSpecLLM(BaseModel):
    """
    Edge specification for LLM generation - no ID field.
    
    Separation of concerns:
    - Data flow is defined by edges.
    - Routing information lives ONLY on edges that originate from a DECISION node.
    """
    source: str = Field(description="Label of the source node.")
    target: str = Field(description="Label of the target node.")
    sourceHandle: str | None = Field(
        default=None,
        description="Optional named output port on the source (if your executor uses ports)."
    )
    targetHandle: str | None = Field(
        default=None,
        description="Optional named input port on the target (if your executor uses ports)."
    )
    route_index: int | None = Field(
        default=None,
        ge=0,
        description="REQUIRED only when 'source' is a DECISION node. Index of the routed branch (0..N)."
    )
    route_label: str | None = Field(
        default=None,
        description="OPTIONAL human-friendly name for the branch (e.g., 'buy', 'sell')."
    )


class EdgeSpec(BaseModel):
    """React Flow compatible edge specification with deterministic ID."""
    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None
    data: EdgeData = Field(default_factory=EdgeData)


# -----------------------------
# Workflow specs
# -----------------------------

class WorkflowSpecLLM(BaseModel):
    """
    WorkflowSpecLLM - The workflow specification you must generate.
    
    NODE TYPES (strict separation of concerns):
    1. data_source: Collects user input ONLY. NEVER calls tools. Config MUST have message + default_value. Use only as inputs to Agents. 
    2. agent: Executes tasks with tools. Can have SLA for enforcement.
    3. decision: Routes workflow using 'routing_gate'. MUST have routing_gate in tools + SLA enforcement.
    
    ROUTING RULES:
    - Decision nodes MUST use routing_gate(data=input, route_index=N, route_name=optional)
    - Edges FROM decision nodes MUST have route_index (0, 1, 2...)
    - Edges from non-decision nodes MUST NOT have route_index
    - routing_gate returns the route_index to control which branch executes
    
    CHAT-ONLY MODE:
    - When user asks questions/chat: Set nodes=null, edges=null, use reasoning for response
    - When user wants workflow: Populate all fields with valid workflow
    - You can think in terms of DAGs (Directed Acyclic Graphs) and understand how to build them and can explain to user what tools, what patterns to generate, etc. 
    
    CRITICAL:
    - Every data_source needs config with message AND default_value
    - Every decision needs routing_gate tool AND SLA with final_tool_must_be="routing_gate"
    - Route indices start at 0 and increment (0=first branch, 1=second, etc.)
    """
    reasoning: str = Field(
        description="Explain design decisions as a chat response, or just chat with user using this field."
    )
    title: str = Field(
        description="Short workflow title. Use null for chat-only responses."
    )
    description: str = Field(
        description="One-sentence workflow description. Use null for chat-only responses."
    )
    nodes: list[NodeSpecLLM] | None = Field(
        default=None,
        description="Workflow nodes. Use null for chat-only responses."
    )
    edges: list[EdgeSpecLLM] | None = Field(
        default=None,
        description="Workflow edges. Use null for chat-only responses."
    )
    
    @model_validator(mode="after")
    def _global_invariants(self) -> WorkflowSpecLLM:
        # chat-only is allowed
        if self.nodes is None and self.edges is None:
            return self
        
        # must have nodes if edges are provided
        if not self.nodes:
            raise ValueError("WorkflowSpecLLM.nodes is required when creating a workflow.")
        
        # Build node-type lookup by label
        label_to_type: dict[str, str] = {}
        for n in self.nodes:
            label_to_type[n.label] = n.type  # type: ignore[attr-defined]
        
        # Enforce edge routing rules
        for e in self.edges or []:
            src_type = label_to_type.get(e.source)
            if src_type == "decision":
                if e.route_index is None:
                    raise ValueError(
                        f"Edge from decision '{e.source}' to '{e.target}' MUST include route_index (0..N)."
                    )
            else:
                # Non-decision sources must not carry routing metadata
                if e.route_index is not None:
                    raise ValueError(
                        f"Edge from non-decision '{e.source}' MUST NOT include route_index."
                    )
        
        return self


class WorkflowSpec(BaseModel):
    """
    Immutable definition of a DAG that the execution engine can run
    and the FE can render with React Flow.
    
    This is the clean output from the LLM with deterministic IDs added.
    System concerns (auth, resources, etc.) are added during conversion.
    """
    id: UUID
    rev: int
    reasoning: str
    title: str
    description: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    metadata: dict = Field(default_factory=dict)  # tags, owner, created_at
    
    @classmethod
    def from_llm_spec(cls, llm_spec: WorkflowSpecLLM, workflow_id: UUID | None = None, rev: int = 1) -> WorkflowSpec:
        """Convert LLM-generated spec to final spec with deterministic IDs."""
        # Generate deterministic node IDs based on type and order
        nodes = []
        node_type_counters = {}
        label_to_id_map = {}  # Map node labels to deterministic IDs
        
        for llm_node in llm_spec.nodes:
            node_type = llm_node.type
            counter = node_type_counters.get(node_type, 0) + 1
            node_type_counters[node_type] = counter
            
            # Create deterministic ID
            node_id = f"{node_type}_{counter}"
            label_to_id_map[llm_node.label] = node_id
            
            # Generate meaningful label if LLM provided generic one
            if llm_node.label and not llm_node.label.startswith(f"{node_type}_"):
                # LLM provided a meaningful label, use it
                node_label = llm_node.label
            else:
                # Generate meaningful label based on node type and data
                if node_type == "data_source":
                    source_name = getattr(llm_node.data, 'source_name', 'Unknown Source')
                    node_label = f"{source_name.replace('_', ' ').title()} Input"
                elif node_type == "agent":
                    # Try to extract meaningful name from agent instructions
                    instructions = getattr(llm_node.data, 'agent_instructions', '')
                    if instructions:
                        # Extract first few words as label
                        words = instructions.split()[:3]
                        node_label = ' '.join(words).title()
                        if len(node_label) > 30:
                            node_label = node_label[:27] + "..."
                    else:
                        node_label = f"Agent {counter}"
                elif node_type == "decision":
                    node_label = f"Decision Point {counter}"
                elif node_type == "workflow_call":
                    workflow_id = getattr(llm_node.data, 'workflow_id', 'Unknown')
                    node_label = f"Call {workflow_id}"
                else:
                    node_label = f"{node_type.replace('_', ' ').title()} {counter}"
            
            # Create the appropriate node type with ID
            node_data = {
                "id": node_id,
                "type": node_type,
                "label": node_label,
                "data": llm_node.data,
                "position": None,
                "runtime": {},
            }
            
            # Add SLA if it exists in the data
            if hasattr(llm_node.data, 'sla') and llm_node.data.sla:
                node_data["sla"] = llm_node.data.sla
            
            # Create the specific node type
            if node_type == "data_source":
                nodes.append(DataSourceNode(**node_data))
            elif node_type == "agent":
                nodes.append(AgentNode(**node_data))
            elif node_type == "decision":
                nodes.append(DecisionNode(**node_data))
            elif node_type == "workflow_call":
                nodes.append(WorkflowCallNode(**node_data))
        
        # Generate edges with deterministic IDs
        edges = []
        for i, llm_edge in enumerate(llm_spec.edges or []):
            # Map source/target labels to IDs
            source_id = label_to_id_map.get(llm_edge.source, llm_edge.source)
            target_id = label_to_id_map.get(llm_edge.target, llm_edge.target)
            
            # Create EdgeData with routing info if present
            edge_data = EdgeData()
            if llm_edge.route_index is not None:
                edge_data.route_index = llm_edge.route_index
            if llm_edge.route_label:
                edge_data.route_label = llm_edge.route_label
            
            edge = EdgeSpec(
                id=f"e_{source_id}_{target_id}_{i}",
                source=source_id,
                target=target_id,
                sourceHandle=llm_edge.sourceHandle,
                targetHandle=llm_edge.targetHandle,
                data=edge_data
            )
            edges.append(edge)
        
        return cls(
            id=workflow_id or uuid4(),
            rev=rev,
            reasoning=llm_spec.reasoning or "",
            title=llm_spec.title or "Untitled Workflow",
            description=llm_spec.description or "",
            nodes=nodes,
            edges=edges,
            metadata={}
        )
    
    def to_llm_prompt(self) -> str:
        """Convert workflow to LLM-friendly structured text."""
        lines = []
        lines.append(f"Workflow: {self.title}")
        lines.append(f"Description: {self.description}")
        lines.append("")
        
        lines.append("Nodes:")
        for node in self.nodes:
            lines.append(f"  - {node.label} ({node.type})")
            if hasattr(node.data, 'agent_instructions'):
                lines.append(f"    Instructions: {node.data.agent_instructions}")
            if hasattr(node.data, 'tools') and node.data.tools:
                lines.append(f"    Tools: {', '.join(node.data.tools)}")
            if hasattr(node, 'sla') and node.sla:
                lines.append(f"    SLA: min_calls={node.sla.min_tool_calls}, required={node.sla.required_tools}")
        
        lines.append("")
        lines.append("Edges:")
        for edge in self.edges:
            source_label = next((n.label for n in self.nodes if n.id == edge.source), edge.source)
            target_label = next((n.label for n in self.nodes if n.id == edge.target), edge.target)
            if edge.data.route_index is not None:
                lines.append(f"  - {source_label} → {target_label} (route {edge.data.route_index}: {edge.data.route_label or 'unlabeled'})")
            else:
                lines.append(f"  - {source_label} → {target_label}")
        
        return "\n".join(lines)
    
    def to_yaml(self) -> str:
        """Export workflow as YAML."""
        import yaml
        # Use centralized serializer to ensure all nested Pydantic models
        # (including tools lists) are preserved correctly.
        from iointel.src.utilities.conversion_utils import ConversionUtils
        
        data = {
            'id': str(self.id),
            'rev': self.rev,
            'title': self.title,
            'description': self.description,
            'reasoning': self.reasoning,
            'nodes': [ConversionUtils.to_jsonable(node) for node in self.nodes],
            'edges': [ConversionUtils.to_jsonable(edge) for edge in self.edges],
            'metadata': self.metadata
        }
        return yaml.dump(data, default_flow_style=False)
    
    def to_workflow_definition(self) -> dict:
        """Convert to executable workflow definition format."""
        from iointel.src.utilities.conversion_utils import ConversionUtils
        
        return {
            'id': str(self.id),
            'rev': self.rev,
            'title': self.title,
            'description': self.description,
            'reasoning': self.reasoning,
            'nodes': [ConversionUtils.to_jsonable(node) for node in self.nodes],
            'edges': [ConversionUtils.to_jsonable(edge) for edge in self.edges],
            'metadata': self.metadata
        }
    
    def validate_structure(self, validation_catalog: dict[str, Any] | None = None) -> list[str]:
        """Validate the workflow structure and return any issues."""
        issues = []
        
        # Check all nodes have unique IDs
        node_ids = [node.id for node in self.nodes]
        node_id_set = set(node_ids)
        if len(node_id_set) != len(node_ids):
            issues.append("Duplicate node IDs found; they must be unique")
        
        # Check all edges reference existing nodes
        for edge in self.edges:
            if edge.source not in node_id_set:
                issues.append(f"Edge references unknown source: {edge.source}")
            if edge.target not in node_id_set:
                issues.append(f"Edge references unknown target: {edge.target}")
        
        # Check for orphaned nodes (no edges) - only if more than 1 node
        if len(self.nodes) > 1:
            connected_nodes = set()
            for edge in self.edges:
                connected_nodes.add(edge.source)
                connected_nodes.add(edge.target)
            
            orphaned = set(node_ids) - connected_nodes
            if orphaned:
                issues.append(f"Orphaned nodes with no connections: {orphaned}")
        
        # Validate node-specific requirements
        if validation_catalog:
            for node in self.nodes:
                node_issues = self._validate_node(node, validation_catalog)
                issues.extend(node_issues)
        
        # Validate routing consistency for decision nodes
        routing_issues = self._validate_routing_consistency()
        issues.extend(routing_issues)
        
        return issues
    
    def _validate_node(self, node: NodeSpec, validation_catalog: dict[str, Any]) -> list[str]:
        """Validate a single node against the validation catalog."""
        issues = []
        
        if node.type == "data_source":
            # Validate data source
            if isinstance(node.data, DataSourceData):
                if node.data.source_name not in validation_catalog:
                    issues.append(f"🚨 HALLUCINATION: Node '{node.label}' uses non-existent data source '{node.data.source_name}'")
                if not node.data.config:
                    issues.append(f"🚨 EMPTY CONFIG: Node '{node.label}' has empty config - MUST include message and default_value")
                elif not isinstance(node.data.config, DataSourceConfig):
                    issues.append(f"🚨 INVALID CONFIG: Node '{node.label}' config must be DataSourceConfig with message and default_value")
        
        elif node.type in ["agent", "decision"]:
            # Validate agent/decision nodes
            if isinstance(node.data, (AgentConfig, DecisionConfig)):
                # Check tools exist in catalog
                for tool in node.data.tools:
                    if tool not in validation_catalog:
                        issues.append(f"🚨 HALLUCINATION: Node '{node.label}' references non-existent tool '{tool}'")
                
                # Decision nodes must have routing tool (routing_gate only)
                if node.type == "decision":
                    if "routing_gate" not in node.data.tools:
                        issues.append(f"Decision node '{node.label}' must include 'routing_gate' in tools")
                    if not node.data.sla or node.data.sla.final_tool_must_be != "routing_gate":
                        issues.append(f"Decision node '{node.label}' must have SLA with final_tool_must_be='routing_gate'")
        
        return issues
    
    def _validate_routing_consistency(self) -> list[str]:
        """Validate routing consistency for decision nodes."""
        issues = []
        
        # Find all decision nodes
        decision_nodes = {node.id: node for node in self.nodes if node.type == "decision"}
        
        for decision_id, decision_node in decision_nodes.items():
            # Find all edges from this decision node
            outgoing_edges = [e for e in self.edges if e.source == decision_id]
            
            if not outgoing_edges:
                issues.append(f"Decision node '{decision_node.label}' has no outgoing edges")
                continue
            
            # Check route indices
            route_indices = []
            for edge in outgoing_edges:
                if edge.data.route_index is None:
                    issues.append(f"Edge from decision '{decision_node.label}' to '{edge.target}' missing route_index")
                else:
                    route_indices.append(edge.data.route_index)
            
            # Check for sequential indices starting from 0
            if route_indices:
                route_indices.sort()
                expected = list(range(len(route_indices)))
                if route_indices != expected:
                    issues.append(f"Decision node '{decision_node.label}' has non-sequential route indices: {route_indices}")
        
        return issues


# -----------------------------
# Execution tracking models
# -----------------------------

class ArtifactRef(BaseModel):
    """Reference to an artifact produced by workflow execution."""
    artifact_id: UUID
    uri: str
    mime: str


class NodeRunSummary(BaseModel):
    """Summary of a single node execution."""
    node_id: str
    status: Literal["success", "failed", "skipped"]
    started_at: str
    finished_at: str
    result_preview: str | None = Field(None, description="first 200 chars / head of CSV")
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None


class WorkflowRunSummary(BaseModel):
    """Summary of a complete workflow execution."""
    workflow_id: UUID
    run_id: UUID
    status: Literal["success", "failed", "partial"]
    started_at: str
    finished_at: str
    node_summaries: list[NodeRunSummary]
    total_duration_seconds: float | None = None
    metadata: dict = Field(default_factory=dict)


# -----------------------------
# Backward compatibility aliases
# -----------------------------

# No backward compatibility - clean break with new structure