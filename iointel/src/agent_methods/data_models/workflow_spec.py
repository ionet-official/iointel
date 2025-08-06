# VALID NODE TYPES: 'data_source', 'agent', 'decision', 'workflow_call'
from typing import Literal, Optional, List, Dict, Set
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from .data_source_registry import ValidDataSourceName

# Centralized routing tools definition
ROUTING_TOOLS = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']


class TestResult(BaseModel):
    """Result of running a test against a workflow."""
    test_id: str
    test_name: str
    passed: bool
    executed_at: datetime
    execution_details: Optional[Dict] = None
    error_message: Optional[str] = None


class TestAlignment(BaseModel):
    """Test alignment metadata for workflows."""
    test_ids: Set[str] = Field(default_factory=set, description="Test IDs that validate this workflow")
    test_results: List[TestResult] = Field(default_factory=list, description="Historical test results")
    last_validated: Optional[datetime] = None
    validation_status: Literal["untested", "passing", "failing", "mixed"] = "untested"
    production_ready: bool = Field(default=False, description="True if all critical tests pass")
    
    def add_test_result(self, result: TestResult):
        """Add a new test result and update validation status."""
        self.test_results.append(result)
        self.test_ids.add(result.test_id)
        self.last_validated = result.executed_at
        self._update_validation_status()
    
    def _update_validation_status(self):
        """Update validation status based on latest test results."""
        if not self.test_results:
            self.validation_status = "untested"
            self.production_ready = False
            return
        
        # Get latest result for each test
        latest_results = {}
        for result in self.test_results:
            if result.test_id not in latest_results or result.executed_at > latest_results[result.test_id].executed_at:
                latest_results[result.test_id] = result
        
        passed = sum(1 for r in latest_results.values() if r.passed)
        total = len(latest_results)
        
        if passed == total:
            self.validation_status = "passing"
            self.production_ready = True
        elif passed == 0:
            self.validation_status = "failing"
            self.production_ready = False
        else:
            self.validation_status = "mixed"
            self.production_ready = False



class SLARequirements(BaseModel):
    """
    SLA requirements for an agent based on its available tools.
    Using BaseModel for consistency with node_execution_wrapper.
    """
    tool_usage_required: bool = Field(default=True, description="Whether the agent must use at least one tool")
    required_tools: List[str] = Field(default_factory=list, description="List of tools that must be used by the agent")
    final_tool_must_be: Optional[str] = Field(default=None, description="Tool that must be called last, if any")
    min_tool_calls: int = Field(default=1, description="Minimum number of tool calls required")
    max_retries: int = Field(default=2, description="Maximum number of retries for the node. No greater than 3.")
    timeout_seconds: int = Field(default=120, description="Timeout for the node. No greater than 300.")
    enforce_usage: bool = Field(default=True, description="Whether SLA enforcement should be applied. If true, the node must use at least one tool.")

class NodeData(BaseModel):
    """Data structure for React Flow node configuration - LLM generated only."""
    # Core configuration that LLM understands
    config: Optional[Dict] = Field(None, description="Tool/agent parameters (e.g., args). Required for data_source and most agent nodes.")
    
    # Data flow ports
    ins: List[str] = Field(default_factory=list, description="Input port names (e.g., 'data', 'query', 'config')")
    outs: List[str] = Field(default_factory=list, description="Output port names (e.g., 'result', 'error', 'status')")
    
    # Execution behavior for multiple dependencies
    execution_mode: Literal["consolidate", "for_each"] = Field(
        "consolidate",
        description="How to handle multiple dependencies: "
                   "consolidate (wait for ALL dependencies, run once with consolidated inputs - cannot be used downstream of decision gates), "
                   "for_each (run separately for each dependency that completes - use downstream of decision gates)"
    )
    
    # Optional fields the LLM might specify
    source_name: Optional[ValidDataSourceName] = Field(None, description="Name of the data source from registry (for data_source nodes only)")
    agent_instructions: Optional[str] = Field(None, description="Instructions for agent (for agent nodes)")
    tools: Optional[List[str]] = Field(None, description="List of tool names available to agent (for agent nodes)")
    workflow_id: Optional[str] = Field(None, description="ID of workflow to call (for workflow_call nodes)")
    model: Optional[Literal["gpt-4o", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"]] = Field("gpt-4o", description="Model to use for agent nodes")
    sla: Optional[SLARequirements] = None


class NodeSpecLLM(BaseModel):
    """Node specification for LLM generation - no ID field."""
    type: Literal["data_source", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}
    sla: Optional[SLARequirements] = Field(None, description="SLA requirements for this node from WorkflowPlanner")


class NodeSpec(BaseModel):
    """React Flow compatible node specification with deterministic ID."""
    id: str
    type: Literal["data_source", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}
    sla: Optional[SLARequirements] = Field(None, description="SLA requirements for this node from WorkflowPlanner")


class EdgeData(BaseModel):
    """Data structure for React Flow edge configuration."""
    route_index: Optional[int] = Field(None, description="Index of the route (0, 1, 2...) for decision routing")
    route_label: Optional[str] = Field(None, description="Human-readable route name (e.g. 'buy', 'sell', 'hold')")
    # Legacy support - will be deprecated
    condition: Optional[str] = Field(None, description="Legacy condition string - use route_index instead")


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
    For chat-only responses, set nodes and edges to null.
    """
    reasoning: str = Field(default="", description="""Your engaging thought process about workflow design and/or response to the user! For refinements: acknowledge SPECIFIC changes requested and explain how you're implementing them. 
For tool listings: organize by category with emojis, highlight capabilities, suggest use cases. 
For workflows: explain your design decisions. Think in the super set, then break down into nodes and wire it up with edges.
                                
REFINEMENT EXAMPLES:
- "Change tools to X" → "I've updated the agent to use only X tools as requested"
- "Remove Y tool" → "I've removed Y tool from the agent's available tools"
- "Add SLA for Z" → "I've added SLA enforcement requiring Z tool usage"
- Always be SPECIFIC about what you changed!""")
    title: Optional[str] = Field(None, description="Workflow title. Use null for chat-only responses.")
    description: Optional[str] = Field(None, description="Workflow description. Use null for chat-only responses.")
    nodes: Optional[List[NodeSpecLLM]] = Field(None, description="Workflow nodes. Use null for chat-only responses to preserve previous DAG.")
    edges: Optional[List[EdgeSpecLLM]] = Field(None, description="Workflow edges. Use null for chat-only responses to preserve previous DAG.")


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
            if llm_node.type == "data_source" and llm_node.data.source_name == "user_input":
                node_id = f"user_input_{counter}"
            else:
                node_id = f"{node_type}_{counter}"
            
            # Map node label to deterministic ID for edge mapping
            label_to_id_map[llm_node.label] = node_id
            
            # Use LLM-specified SLA (either at node level or data level)
            final_sla = llm_node.sla or getattr(llm_node.data, 'sla', None)
            
            nodes.append(NodeSpec(
                id=node_id,
                type=llm_node.type,
                label=llm_node.label,
                data=llm_node.data,
                position=llm_node.position,
                runtime=llm_node.runtime,
                sla=final_sla  # Store SLA at NodeSpec level as authoritative source
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
    
    def to_llm_prompt(self) -> str:
        """
        Convert to structured, LLM-friendly representation.
        
        This is the hermetic single source of truth for workflow representation.
        Includes topology, SLAs, routing logic, and all critical information.
        """
        lines = []
        
        # Header with metadata
        lines.append("📋 WORKFLOW SPECIFICATION")
        lines.append("=" * 50)
        lines.append(f"Title: {self.title}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Reasoning: {self.reasoning}")
        lines.append(f"ID: {self.id}")
        lines.append(f"Version: {self.rev}")
        lines.append("")
        
        # Topology overview
        lines.append("🏗️ TOPOLOGY OVERVIEW")
        lines.append("-" * 25)
        lines.append(f"Total Nodes: {len(self.nodes)}")
        lines.append(f"Total Edges: {len(self.edges)}")
        
        # Categorize nodes
        node_types = {}
        decision_nodes = []
        sla_nodes = []
        
        for node in self.nodes:
            node_type = node.type
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Check for decision nodes  
            if self._is_decision_node(node):
                decision_nodes.append(node.id)
            
            # Check for SLA enforcement
            if self._has_sla_enforcement(node):
                sla_nodes.append(node.id)
        
        lines.append(f"Node Types: {dict(node_types)}")
        if decision_nodes:
            lines.append(f"Decision Nodes: {decision_nodes}")
        if sla_nodes:
            lines.append(f"SLA-Enforced Nodes: {sla_nodes}")
        lines.append("")
        
        # Detailed node specifications
        lines.append("🔗 NODE SPECIFICATIONS")
        lines.append("-" * 30)
        
        for node in self.nodes:
            lines.append(f"Node ID: {node.id}")
            lines.append(f"  Label: {node.label}")
            lines.append(f"  Type: {node.type}")
            
            # Node data details
            if hasattr(node.data, 'agent_instructions') and node.data.agent_instructions:
                lines.append(f"  Instructions: {node.data.agent_instructions}")
            
            if hasattr(node.data, 'tools') and node.data.tools:
                lines.append(f"  Tools: {node.data.tools}")
            
            if hasattr(node.data, 'source_name') and node.data.source_name:
                lines.append(f"  Data Source: {node.data.source_name}")
                if hasattr(node.data, 'config') and node.data.config:
                    lines.append(f"  Config: {node.data.config}")
            
            # SLA information
            if hasattr(node, 'sla') and node.sla:
                lines.append(f"  SLA:")
                if hasattr(node.sla, 'enforce_usage') and node.sla.enforce_usage:
                    lines.append(f"    Enforce Usage: {node.sla.enforce_usage}")
                if hasattr(node.sla, 'required_tools') and node.sla.required_tools:
                    lines.append(f"    Required Tools: {node.sla.required_tools}")
                if hasattr(node.sla, 'final_tool_must_be') and node.sla.final_tool_must_be:
                    lines.append(f"    Final Tool Call *Must Be*: {node.sla.final_tool_must_be}")
            
            lines.append("")
        
        # Edge specifications with routing logic
        lines.append("🎯 EDGE SPECIFICATIONS & ROUTING")
        lines.append("-" * 40)
        
        for edge in self.edges:
            lines.append(f"Edge: {edge.source} → {edge.target}")
            if edge.sourceHandle:
                lines.append(f"  Source Handle: {edge.sourceHandle}")
            if edge.targetHandle:
                lines.append(f"  Target Handle: {edge.targetHandle}")
            
            # Routing logic
            if edge.data and hasattr(edge.data, 'condition') and edge.data.condition:
                lines.append(f"  Condition: {edge.data.condition}")
            
            if edge.data and hasattr(edge.data, 'route_index') and edge.data.route_index is not None:
                lines.append(f"  Route Index: {edge.data.route_index}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _is_decision_node(self, node) -> bool:
        """Check if a node is a decision/routing node."""
        return (node.type == 'decision' or 
                (hasattr(node.data, 'tools') and node.data.tools and 
                 any(tool in ['conditional_gate', 'routing_gate'] for tool in node.data.tools)))
    
    def _has_sla_enforcement(self, node) -> bool:
        """Check if a node has SLA enforcement."""
        return (hasattr(node, 'sla') and node.sla and 
                hasattr(node.sla, 'enforce_usage') and node.sla.enforce_usage)
    
    # DEPRECATED: Node formatting methods moved to centralized conversion_utils
    # These will be removed after full migration
    
    def validate_structure(self, tool_catalog: dict = None) -> List[str]:
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
        
        # Check for orphaned nodes (no edges)
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        orphaned = set(node_ids) - connected_nodes
        if orphaned and len(self.nodes) > 1:
            issues.append(f"Orphaned nodes with no connections: {orphaned}")
        
        # Add routing consistency validation
        routing_issues = self.validate_routing_consistency()
        issues.extend(routing_issues)
        
        # Validate node-type specific requirements
        for node in self.nodes:
            # Validate SLA configuration if present
            self._validate_sla_configuration(node, issues)
            
            if node.type == "data_source":
                self._validate_data_source_node(node, issues, tool_catalog)
            
            elif node.type == "agent":
                self._validate_agent_node(node, issues)
            
            elif node.type == "workflow_call":
                self._validate_workflow_call_node(node, issues)
            
            elif node.type == "decision":
                self._validate_decision_node(node, issues, tool_catalog)
        return issues

    def _validate_sla_configuration(self, node, issues):
        """Validate SLA configuration for a specific node."""
        if not hasattr(node, 'sla') or not node.sla:
            return
        
        sla = node.sla
        # Validate SLA references actual tools available to the node
        if sla.required_tools:
            available_tools = set(node.data.tools or [])
            missing_sla_tools = set(sla.required_tools) - available_tools
            if missing_sla_tools:
                issues.append(f"🚨 SLA MISCONFIGURATION: Node '{node.id}' SLA requires tools not available to node: {sorted(missing_sla_tools)}. Available: {sorted(available_tools)}")
        
        # Validate final tool requirement
        if sla.final_tool_must_be and sla.final_tool_must_be not in (node.data.tools or []):
            issues.append(f"🚨 SLA MISCONFIGURATION: Node '{node.id}' SLA requires final tool '{sla.final_tool_must_be}' not available to node")
        
        # Validate logical consistency
        if sla.enforce_usage and not sla.tool_usage_required and not sla.required_tools:
            issues.append(f"⚠️ SLA WARNING: Node '{node.id}' has enforce_usage=True but no tool requirements specified")

    def _validate_data_source_node(self, node, issues, tool_catalog):
        """Validate data source node configuration."""
        if not node.data.source_name:
            issues.append(f"Data source node '{node.id}' ({node.label}) missing required 'source_name'")
        elif tool_catalog and node.data.source_name not in tool_catalog:
            issues.append(f"🚨 SOURCE HALLUCINATION: Node '{node.id}' uses non-existent source '{node.data.source_name}'. Available sources: {sorted(tool_catalog.keys())}")
        elif tool_catalog and node.data.source_name in tool_catalog:
            # Validate source parameters
            tool_info = tool_catalog[node.data.source_name]
            required_params = tool_info.get("required_parameters", [])
            tool_info.get("parameters", {})
            config_params = set(node.data.config.keys()) if node.data.config else set()
            
            # Check for missing required parameters - NO AUTO-HEALING, FAIL FAST
            missing_params = set(required_params) - config_params
            if missing_params:
                issues.append(f"🚨 MISSING PARAMETERS: Data source node '{node.id}' ({node.data.source_name}) missing required parameters: {sorted(missing_params)}. Config has: {sorted(config_params)}")
            
            # Check for None/empty config - NO AUTO-HEALING, FAIL FAST  
            if required_params and (not node.data.config):
                issues.append(f"🚨 EMPTY CONFIG: Data source node '{node.id}' ({node.data.source_name}) has None/empty config but requires parameters: {sorted(required_params)}")

    def _validate_agent_node(self, node, issues):
        """Validate agent node configuration."""
        if not node.data.agent_instructions:
            issues.append(f"Agent node '{node.id}' ({node.label}) missing required 'agent_instructions'")

    def _validate_workflow_call_node(self, node, issues):
        """Validate workflow call node configuration."""
        if not node.data.workflow_id:
            issues.append(f"Workflow call node '{node.id}' ({node.label}) missing required 'workflow_id'")

    def _validate_decision_node(self, node, issues, tool_catalog):
        """Validate decision node configuration."""
        if hasattr(node.data, 'tools') and node.data.tools:
            # Check if decision node tools exist in catalog
            for tool in node.data.tools:
                if tool_catalog and tool not in tool_catalog:
                    issues.append(f"🚨 TOOL HALLUCINATION: Decision node '{node.id}' uses non-existent tool '{tool}'. Available tools: {sorted(tool_catalog.keys())}")

    def validate_routing_consistency(self, mode: str = "strict") -> List[str]:
        """
        Validate routing consistency by checking edges are properly wired:
        1. Decision/routing nodes have conditional outgoing edges
        2. No dangling conditional edges 
        3. All conditional edges have matching targets
        4. SEMANTIC: Route names match between tools and edge conditions
        """
        issues = []
        
        # Build edge mapping
        outgoing_edges = {}
        incoming_edges = {}
        
        for edge in self.edges:
            if edge.source not in outgoing_edges:
                outgoing_edges[edge.source] = []
            outgoing_edges[edge.source].append(edge)
            
            if edge.target not in incoming_edges:
                incoming_edges[edge.target] = []
            incoming_edges[edge.target].append(edge)
        
        for node in self.nodes:
            has_routing_tool = False
            
            # Check if node uses routing tools
            if (node.type == "decision" or node.type == "agent") and node.data.tools:
                has_routing_tool = any(tool in ROUTING_TOOLS for tool in node.data.tools)
            
            if has_routing_tool:
                node_edges = outgoing_edges.get(node.id, [])
                
                # Must have outgoing edges
                if not node_edges:
                    issues.append(f"🚨 DANGLING ROUTING NODE: '{node.id}' ({node.label}) uses routing tools but has no outgoing edges")
                    continue
                
                # Check for routing vs unconditional edges (using route_index)
                routing_edges = [e for e in node_edges if e.data and e.data.route_index is not None]
                #[e for e in node_edges if not (e.data and e.data.route_index is not None)]
                
                if not routing_edges:
                    issues.append(f"🚨 MISSING ROUTE_INDEX: Routing node '{node.id}' ({node.label}) has outgoing edges but none have route_index set")
                
                # All targets must exist
                for edge in node_edges:
                    target_exists = any(n.id == edge.target for n in self.nodes)
                    if not target_exists:
                        issues.append(f"🚨 BROKEN EDGE: Edge '{edge.id}' points to non-existent target '{edge.target}'")
        
        # Check for orphaned routing edges (edges with route_index from non-routing nodes)
        for edge in self.edges:
            if edge.data and edge.data.route_index is not None:
                source_node = next((n for n in self.nodes if n.id == edge.source), None)
                if source_node:
                    has_routing_tool = False
                    
                    if (source_node.type == "decision" or source_node.type == "agent") and source_node.data.tools:
                        has_routing_tool = any(tool in ROUTING_TOOLS for tool in source_node.data.tools)
                    
                    if not has_routing_tool:
                        issues.append(f"⚠️ ORPHANED ROUTE_INDEX: Edge '{edge.id}' has route_index '{edge.data.route_index}' but source node '{edge.source}' doesn't use routing tools")
        
        # Check for unreachable nodes (no incoming edges)
        all_targets = {edge.target for edge in self.edges}
        all_sources = {edge.source for edge in self.edges}
        
        for node in self.nodes:
            if node.id not in all_targets and node.id in all_sources:
                # This is a root node (has outgoing but no incoming) - OK
                continue
            elif node.id not in all_targets and node.id not in all_sources:
                # Isolated node - only flag as error if there are multiple nodes
                if len(self.nodes) > 1:
                    # Multiple nodes but this one is isolated - that's an error
                    issues.append(f"🚨 ISOLATED NODE: Node '{node.id}' ({node.label}) has no connections")
                # Single isolated nodes are OK (e.g., standalone agent workflows)
        
        # 4. SEMANTIC ROUTE VALIDATION: Check route names match between tools and edges
        for node in self.nodes:
            has_routing_tools = False
            
            # Check both agent nodes with tools and decision nodes with tools
            if (node.type == 'agent' or node.type == 'decision') and node.data.tools:
                has_routing_tools = any(tool in ROUTING_TOOLS for tool in node.data.tools)
                
            if has_routing_tools and node.id in outgoing_edges:
                conditional_edges = [e for e in outgoing_edges[node.id] if e.data.condition]
                
                # Check for incompatible condition patterns
                for edge in conditional_edges:
                    condition = edge.data.condition or ""
                    
                    # CRITICAL: Check for decision== patterns when routing tools are used
                    if "decision ==" in condition and "routed_to ==" not in condition:
                        issues.append(f"🚨 ROUTING MISMATCH: Node '{node.id}' uses routing tools ({ROUTING_TOOLS}) but edge conditions use 'decision ==' pattern. DAG executor expects 'routed_to ==' pattern. Edge condition: '{condition}'")
                    
                    # Check for other incompatible patterns
                    if "action ==" in condition and "routed_to ==" not in condition:
                        issues.append(f"🚨 ROUTING MISMATCH: Node '{node.id}' uses routing tools but edge conditions use 'action ==' pattern. DAG executor expects 'routed_to ==' pattern. Edge condition: '{condition}'")
                    
                    # Check conditional_gate custom route extraction from instructions
                    if node.data.tools and 'conditional_gate' in node.data.tools and node.data.agent_instructions:
                        # Try to extract custom routes from agent instructions
                        instructions = node.data.agent_instructions or ""
                        if '"route":' in instructions:
                            # Simple regex-like extraction of custom routes
                            import re
                            route_matches = re.findall(r'"route":\s*"([^"]+)"', instructions)
                            if route_matches:
                                instruction_routes = set(route_matches)
                                edge_routes = set()
                                for edge in conditional_edges:
                                    condition = edge.data.condition or ""
                                    if "routed_to ==" in condition:
                                        route_match = condition.split("'")[1] if "'" in condition else None
                                        if route_match:
                                            edge_routes.add(route_match)
                                
                                route_mismatches = instruction_routes - edge_routes
                                if route_mismatches:
                                    issues.append(f"🚨 ROUTE MISMATCH: conditional_gate in '{node.id}' configured with routes {instruction_routes}, but edges expect {edge_routes}")
        
        return issues
    
    # Test Alignment Methods
    def get_test_alignment(self) -> TestAlignment:
        """Get test alignment metadata for this workflow."""
        alignment_data = self.metadata.get('test_alignment', {})
        return TestAlignment(**alignment_data) if alignment_data else TestAlignment()
    
    def add_test_result(self, test_id: str, test_name: str, passed: bool, 
                       execution_details: Optional[Dict] = None, error_message: Optional[str] = None):
        """Add a test result to this workflow's alignment metadata."""
        alignment = self.get_test_alignment()
        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            passed=passed,
            executed_at=datetime.now(),
            execution_details=execution_details,
            error_message=error_message
        )
        alignment.add_test_result(result)
        self.metadata['test_alignment'] = alignment.model_dump()
    
    def is_production_ready(self) -> bool:
        """Check if this workflow is ready for production use."""
        alignment = self.get_test_alignment()
        return alignment.production_ready
    
    def get_validation_status(self) -> Literal["untested", "passing", "failing", "mixed"]:
        """Get the current validation status of this workflow."""
        alignment = self.get_test_alignment()
        return alignment.validation_status
    
    def get_failing_tests(self) -> List[TestResult]:
        """Get list of tests that are currently failing for this workflow."""
        alignment = self.get_test_alignment()
        if not alignment.test_results:
            return []
        
        # Get latest result for each test
        latest_results = {}
        for result in alignment.test_results:
            if result.test_id not in latest_results or result.executed_at > latest_results[result.test_id].executed_at:
                latest_results[result.test_id] = result
        
        return [result for result in latest_results.values() if not result.passed]
    
    def link_to_test(self, test_id: str):
        """Link this workflow to a test without running it."""
        alignment = self.get_test_alignment()
        alignment.test_ids.add(test_id)
        self.metadata['test_alignment'] = alignment.model_dump()
    
    def validate_tools(self, tool_catalog: dict) -> List[str]:
        """Specifically validate that all tools used exist in the catalog."""
        issues = []
        available_tools = set(tool_catalog.keys())
        
        for node in self.nodes:
            if hasattr(node.data, 'source_name') and node.data.source_name:
                if node.data.source_name not in available_tools:
                    issues.append(f"Node '{node.id}' uses (hallucinated) unavailable data source '{node.data.source_name}'")
            elif hasattr(node.data, 'tools') and node.data.tools:
                for tool in node.data.tools:
                    if tool not in available_tools:
                        issues.append(f"Node '{node.id}' uses (hallucinated) unavailable tool '{tool}'")
        
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