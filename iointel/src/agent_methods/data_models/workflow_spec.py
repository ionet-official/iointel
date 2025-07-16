from typing import Literal, Optional, List, Dict
from uuid import UUID
from pydantic import BaseModel, Field


class NodeData(BaseModel):
    """Data structure for React Flow node configuration - LLM generated only."""
    # Core configuration that LLM understands
    config: Dict = Field(default_factory=dict, description="Tool/agent parameters (e.g., query, format)")
    
    # Data flow ports
    ins: List[str] = Field(default_factory=list, description="Input port names (e.g., 'data', 'query', 'config')")
    outs: List[str] = Field(default_factory=list, description="Output port names (e.g., 'result', 'error', 'status')")
    
    # Optional fields the LLM might specify
    tool_name: Optional[str] = Field(None, description="Name of the tool from catalog (for tool nodes)")
    agent_instructions: Optional[str] = Field(None, description="Instructions for agent (for agent nodes)")
    tools: Optional[List[str]] = Field(None, description="List of tool names available to agent (for agent nodes)")
    workflow_id: Optional[str] = Field(None, description="ID of workflow to call (for workflow_call nodes)")
    model: Optional[Literal["gpt-4o", "gpt-4", "gpt-3.5-turbo", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]] = Field("gpt-4o", description="Model to use for agent nodes")


class NodeSpecLLM(BaseModel):
    """Node specification for LLM generation - no ID field."""
    type: Literal["tool", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}


class NodeSpec(BaseModel):
    """React Flow compatible node specification with deterministic ID."""
    id: str
    type: Literal["tool", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}


class EdgeData(BaseModel):
    """Data structure for React Flow edge configuration."""
    condition: Optional[str] = Field(None, description="e.g. \"status=='success'\"")


class EdgeSpecLLM(BaseModel):
    """Edge specification for LLM generation - no ID field."""
    source: str = Field(..., description="Source node label or descriptive name")
    target: str = Field(..., description="Target node label or descriptive name")
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: EdgeData = Field(default_factory=EdgeData)


class EdgeSpec(BaseModel):
    """React Flow compatible edge specification with deterministic ID."""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: EdgeData = Field(default_factory=EdgeData)


class WorkflowSpecLLM(BaseModel):
    """
    Workflow specification for LLM generation - no IDs, system generates them.
    """
    reasoning: str = Field(default="", description="LLM's chat bot response to the user's query about workflow creation, including constraints and limitations or suggestions for improvements.")
    title: str
    description: str = ""
    nodes: List[NodeSpecLLM]
    edges: List[EdgeSpecLLM]


class WorkflowSpec(BaseModel):
    """
    Immutable definition of a DAG that the execution engine can run
    and the FE can render with React Flow.
    
    This is the clean output from the LLM with deterministic IDs added.
    System concerns (auth, resources, etc.) are added during conversion.
    """
    id: UUID
    rev: int
    reasoning: str = Field(default="", description="LLM's chat bot response to the user's query about workflow creation, including constraints and limitations or suggestions for improvements.")
    title: str
    description: str = ""
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    metadata: Dict = Field(default_factory=dict)  # tags, owner, created_at
    
    @classmethod
    def from_llm_spec(cls, llm_spec: WorkflowSpecLLM, workflow_id: UUID = None, rev: int = 1) -> "WorkflowSpec":
        """Convert LLM-generated spec to final spec with deterministic IDs."""
        from uuid import uuid4
        
        # Generate deterministic node IDs based on type and order
        nodes = []
        node_type_counters = {}
        label_to_id_map = {}  # Map node labels to deterministic IDs
        
        for llm_node in llm_spec.nodes:
            node_type = llm_node.type
            counter = node_type_counters.get(node_type, 0) + 1
            node_type_counters[node_type] = counter
            
            # Create deterministic ID: type_counter (e.g., "tool_1", "agent_1", "user_input_1")
            if llm_node.type == "tool" and llm_node.data.tool_name == "user_input":
                node_id = f"user_input_{counter}"
            else:
                node_id = f"{node_type}_{counter}"
            
            # Map node label to deterministic ID for edge mapping
            label_to_id_map[llm_node.label] = node_id
            
            nodes.append(NodeSpec(
                id=node_id,
                type=llm_node.type,
                label=llm_node.label,
                data=llm_node.data,
                position=llm_node.position,
                runtime=llm_node.runtime
            ))
        
        # Generate deterministic edge IDs and map source/target to new node IDs
        edges = []
        for i, llm_edge in enumerate(llm_spec.edges, 1):
            # Map source and target labels to deterministic IDs
            source_id = label_to_id_map.get(llm_edge.source, llm_edge.source)
            target_id = label_to_id_map.get(llm_edge.target, llm_edge.target)
            
            edges.append(EdgeSpec(
                id=f"edge_{i}",
                source=source_id,
                target=target_id,
                sourceHandle=llm_edge.sourceHandle,
                targetHandle=llm_edge.targetHandle,
                data=llm_edge.data
            ))
        
        return cls(
            id=workflow_id or uuid4(),
            rev=rev,
            reasoning=llm_spec.reasoning,
            title=llm_spec.title,
            description=llm_spec.description,
            nodes=nodes,
            edges=edges
        )
    
    def to_workflow_definition(self, **kwargs):
        """Convert to executable WorkflowDefinition format."""
        from ..workflow_converter import spec_to_definition
        return spec_to_definition(self, **kwargs)
    
    def to_yaml(self, **kwargs) -> str:
        """Export as YAML for persistence/sharing."""
        from ..workflow_converter import spec_to_yaml
        return spec_to_yaml(self, **kwargs)
    
    def validate_structure(self, tool_catalog: dict = None) -> List[str]:
        """Validate the workflow structure and return any issues."""
        issues = []
        
        # Check all nodes have unique IDs
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            issues.append("Duplicate node IDs found")
        
        # Check all edges reference existing nodes
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.source not in node_id_set:
                issues.append(f"Edge references unknown source: {edge.source}")
            if edge.target not in node_id_set:
                issues.append(f"Edge references unknown target: {edge.target}")
        
        # Check for orphaned nodes (no edges)
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        orphaned = set(node_ids) - connected_nodes
        if orphaned and len(self.nodes) > 1:
            issues.append(f"Orphaned nodes with no connections: {orphaned}")
        
        # Validate node-type specific requirements
        for node in self.nodes:
            if node.type == "tool":
                if not node.data.tool_name:
                    issues.append(f"Tool node '{node.id}' ({node.label}) missing required 'tool_name'")
                elif tool_catalog and node.data.tool_name not in tool_catalog:
                    issues.append(f"🚨 TOOL HALLUCINATION: Node '{node.id}' uses non-existent tool '{node.data.tool_name}'. Available tools: {sorted(tool_catalog.keys())}")
                elif tool_catalog and node.data.tool_name in tool_catalog:
                    # Validate tool parameters
                    tool_info = tool_catalog[node.data.tool_name]
                    required_params = tool_info.get("required_parameters", [])
                    all_params = tool_info.get("parameters", {})
                    config_params = set(node.data.config.keys())
                    
                    # Check for missing required parameters (only check actually required ones)
                    missing_params = set(required_params) - config_params
                    if missing_params:
                        issues.append(f"🚨 MISSING PARAMETERS: Tool node '{node.id}' ({node.data.tool_name}) missing required parameters: {sorted(missing_params)}. Config has: {sorted(config_params)}")
                    
                    # Check for empty config when parameters are required
                    if required_params and not node.data.config:
                        issues.append(f"🚨 EMPTY CONFIG: Tool node '{node.id}' ({node.data.tool_name}) has empty config but requires parameters: {sorted(required_params)}")
            
            elif node.type == "agent":
                if not node.data.agent_instructions:
                    issues.append(f"Agent node '{node.id}' ({node.label}) missing required 'agent_instructions'")
            
            elif node.type == "workflow_call":
                if not node.data.workflow_id:
                    issues.append(f"Workflow call node '{node.id}' ({node.label}) missing required 'workflow_id'")
            
            elif node.type == "decision" and node.data.tool_name:
                if tool_catalog and node.data.tool_name not in tool_catalog:
                    issues.append(f"🚨 TOOL HALLUCINATION: Decision node '{node.id}' uses non-existent tool '{node.data.tool_name}'. Available tools: {sorted(tool_catalog.keys())}")
        return issues
    
    def validate_tools(self, tool_catalog: dict) -> List[str]:
        """Specifically validate that all tools used exist in the catalog."""
        issues = []
        available_tools = set(tool_catalog.keys())
        
        for node in self.nodes:
            if node.data.tool_name:
                if node.data.tool_name not in available_tools:
                    issues.append(f"Node '{node.id}' uses (hallucinated) unavailable tool '{node.data.tool_name}'")
        
        return issues


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
    result_preview: Optional[str] = Field(None, description="first 200 chars / head of CSV")
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    error_message: Optional[str] = None


class WorkflowRunSummary(BaseModel):
    """Summary of a complete workflow execution."""
    workflow_id: UUID
    run_id: UUID
    status: Literal["success", "failed", "partial"]
    started_at: str
    finished_at: str
    node_summaries: List[NodeRunSummary]
    total_duration_seconds: Optional[float] = None
    metadata: Dict = Field(default_factory=dict)