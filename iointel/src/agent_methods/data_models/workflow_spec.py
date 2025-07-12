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


class NodeSpec(BaseModel):
    """React Flow compatible node specification."""
    id: str
    type: Literal["tool", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}


class EdgeData(BaseModel):
    """Data structure for React Flow edge configuration."""
    condition: Optional[str] = Field(None, description="e.g. \"status=='success'\"")


class EdgeSpec(BaseModel):
    """React Flow compatible edge specification."""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: EdgeData = Field(default_factory=EdgeData)


class WorkflowSpec(BaseModel):
    """
    Immutable definition of a DAG that the execution engine can run
    and the FE can render with React Flow.
    
    This is the clean output from the LLM - minimal and focused on structure.
    System concerns (auth, resources, etc.) are added during conversion.
    """
    id: UUID
    rev: int
    title: str
    description: str = ""
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    metadata: Dict = Field(default_factory=dict)  # tags, owner, created_at
    reasoning: str = Field(default="", description="LLM's chat bot response to the user's query about workflow creation, including constraints and limitations")
    
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
                    required_params = tool_info.get("parameters", {})
                    config_params = set(node.data.config.keys())
                    
                    # Check for missing required parameters
                    missing_params = set(required_params.keys()) - config_params
                    if missing_params:
                        issues.append(f"🚨 MISSING PARAMETERS: Tool node '{node.id}' ({node.data.tool_name}) missing required parameters: {sorted(missing_params)}. Config has: {sorted(config_params)}")
                    
                    # Check for empty config when parameters are required
                    if required_params and not node.data.config:
                        issues.append(f"🚨 EMPTY CONFIG: Tool node '{node.id}' ({node.data.tool_name}) has empty config but requires parameters: {sorted(required_params.keys())}")
            
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
                    issues.append(f"Node '{node.id}' uses unavailable tool '{node.data.tool_name}'")
        
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