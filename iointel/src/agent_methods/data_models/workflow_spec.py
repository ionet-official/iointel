from typing import Literal, Optional, List, Dict
from uuid import UUID
from pydantic import BaseModel, Field



class SLARequirements(BaseModel):
    """
    SLA requirements for an agent based on its available tools.
    Using BaseModel for consistency with node_execution_wrapper.
    """
    tool_usage_required: bool = Field(default=False, description="Whether the agent must use at least one tool")
    required_tools: List[str] = Field(default_factory=list, description="List of tools that must be used by the agent")
    final_tool_must_be: Optional[str] = Field(default=None, description="Tool that must be called last, if any")
    min_tool_calls: int = Field(default=0, description="Minimum number of tool calls required")
    max_retries: int = Field(default=2, description="Maximum number of retries for the node. No greater than 3.")
    timeout_seconds: int = Field(default=120, description="Timeout for the node. No greater than 300.")
    enforce_usage: bool = Field(default=False, description="Whether SLA enforcement should be applied. If true, the node must use at least one tool.")

class NodeData(BaseModel):
    """Data structure for React Flow node configuration - LLM generated only."""
    # Core configuration that LLM understands
    config: Dict = Field(default_factory=dict, description="Tool/agent parameters (e.g., query, format)")
    
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
    tool_name: Optional[str] = Field(None, description="Name of the tool from catalog (for tool nodes)")
    agent_instructions: Optional[str] = Field(None, description="Instructions for agent (for agent nodes)")
    tools: Optional[List[str]] = Field(None, description="List of tool names available to agent (for agent nodes)")
    workflow_id: Optional[str] = Field(None, description="ID of workflow to call (for workflow_call nodes)")
    model: Optional[Literal["gpt-4o", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]] = Field("gpt-4o", description="Model to use for agent nodes")
    sla: Optional[SLARequirements] = Field(
        default=None, 
        description="SLA requirements for the node. Use null for type-based defaults."
    )


class NodeSpecLLM(BaseModel):
    """Node specification for LLM generation - no ID field."""
    type: Literal["tool", "agent", "workflow_call", "decision"]
    label: str
    data: NodeData
    position: Optional[Dict[str, float]] = None  # FE inserts {x,y}
    runtime: Dict = Field(default_factory=dict)  # e.g. {"timeout":30,"retries":1}
    sla: Optional[SLARequirements] = Field(
        None, 
        description="SLA requirements for this node. Use null for type-based defaults: decision nodes require routing tools as final tool, agents are lax by default."
    )


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
    For chat-only responses, set nodes and edges to null.
    """
    reasoning: str = Field(default="", description="Your engaging response to the user! For tool listings: organize by category with emojis, highlight capabilities, suggest use cases. For workflows: explain your design decisions. Be enthusiastic about IO.net's capabilities!")
    title: Optional[str] = Field(None, description="Workflow title. Use null for chat-only responses.")
    description: str = Field(default="", description="Workflow description or chat response message.")
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
    reasoning: str = Field(default="", description="Your engaging response to the user! For tool listings: organize by category with emojis, highlight capabilities, suggest use cases. For workflows: explain your design decisions. Be enthusiastic about IO.net's capabilities!")
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
            
            # Use LLM-specified SLA (either at node level or data level)
            final_sla = llm_node.sla or llm_node.data.sla
                
            # Create node data with merged SLA
            node_data = llm_node.data.model_copy()
            node_data.sla = final_sla
            
            nodes.append(NodeSpec(
                id=node_id,
                type=llm_node.type,
                label=llm_node.label,
                data=node_data,
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
    
    def to_llm_prompt(self) -> str:
        """
        Convert to structured, LLM-friendly representation.
        
        This is the single source of truth for how workflows are presented to LLMs.
        Includes topology, SLAs, routing logic, and all critical information.
        """
        lines = []
        
        # Header with metadata
        lines.append("📋 WORKFLOW SPECIFICATION")
        lines.append("=" * 50)
        lines.append(f"Title: {self.title}")
        lines.append(f"Description: {self.description}")
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
        
        for node_type, count in node_types.items():
            lines.append(f"- {node_type}: {count}")
        
        if decision_nodes:
            lines.append(f"- Decision nodes: {', '.join(decision_nodes)}")
        
        if sla_nodes:
            lines.append(f"- SLA enforced: {', '.join(sla_nodes)}")
        
        lines.append("")
        
        # Node details with SLAs
        lines.append("🔍 NODE SPECIFICATIONS")
        lines.append("-" * 25)
        
        for node in self.nodes:
            lines.extend(self._format_node_details(node))
            lines.append("")
        
        # Edge routing logic
        lines.append("🔀 ROUTING LOGIC")
        lines.append("-" * 20)
        
        if not self.edges:
            lines.append("No routing edges defined (linear execution)")
        else:
            # Group edges by source
            edges_by_source = {}
            for edge in self.edges:
                if edge.source not in edges_by_source:
                    edges_by_source[edge.source] = []
                edges_by_source[edge.source].append(edge)
            
            for source_id, edges in edges_by_source.items():
                source_node = next((n for n in self.nodes if n.id == source_id), None)
                if source_node:
                    lines.append(f"From {source_id} ({source_node.label}):")
                    
                    for edge in edges:
                        target_node = next((n for n in self.nodes if n.id == edge.target), None)
                        condition_str = f" [condition: {edge.data.condition}]" if edge.data.condition else ""
                        target_label = target_node.label if target_node else edge.target
                        lines.append(f"  → {edge.target} ({target_label}){condition_str}")
                    lines.append("")
        
        # Expected execution patterns
        lines.append("⚡ EXPECTED EXECUTION PATTERNS")
        lines.append("-" * 35)
        
        if decision_nodes:
            lines.append("CONDITIONAL ROUTING EXPECTED:")
            lines.append("- Only ONE path should execute based on conditions")
            lines.append("- Other branches should be skipped (not failures)")
            lines.append("- Efficiency = executed_nodes / nodes_on_chosen_path")
            lines.append("")
        
        if sla_nodes:
            lines.append("SLA ENFORCEMENT ACTIVE:")
            for node_id in sla_nodes:
                node = next((n for n in self.nodes if n.id == node_id), None)
                if node and hasattr(node.data, 'sla') and node.data.sla:
                    sla = node.data.sla
                    lines.append(f"- {node_id}: {self._format_sla_requirements(sla)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_node_details(self, node: 'NodeSpec') -> List[str]:
        """Format detailed node information."""
        lines = []
        
        # Node header with type indicator
        node_indicator = "🎯" if self._is_decision_node(node) else "🤖" if node.type == "agent" else "🔧"
        sla_indicator = " [SLA]" if self._has_sla_enforcement(node) else ""
        lines.append(f"{node_indicator} {node.id} - {node.label} ({node.type}){sla_indicator}")
        
        # Instructions/purpose
        if hasattr(node.data, 'agent_instructions') and node.data.agent_instructions:
            lines.append(f"   Purpose: {node.data.agent_instructions[:100]}...")
        elif hasattr(node.data, 'tool_name') and node.data.tool_name:
            lines.append(f"   Tool: {node.data.tool_name}")
        
        # Tools available
        if hasattr(node.data, 'tools') and node.data.tools:
            routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
            tool_list = []
            for tool in node.data.tools:
                if tool in routing_tools:
                    tool_list.append(f"🔀{tool}")  # Routing tool
                else:
                    tool_list.append(f"🔧{tool}")  # Regular tool
            lines.append(f"   Tools: {', '.join(tool_list)}")
        
        # SLA details
        if hasattr(node.data, 'sla') and node.data.sla:
            lines.append(f"   SLA: {self._format_sla_requirements(node.data.sla)}")
        
        # Configuration
        if hasattr(node.data, 'config') and node.data.config:
            lines.append(f"   Config: {node.data.config}")
        
        return lines
    
    def _is_decision_node(self, node: 'NodeSpec') -> bool:
        """Check if node is a decision/routing node."""
        if node.type == "decision":
            return True
        
        if hasattr(node.data, 'tools') and node.data.tools:
            routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
            return any(tool in routing_tools for tool in node.data.tools)
        
        if hasattr(node.data, 'tool_name') and node.data.tool_name:
            routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
            return node.data.tool_name in routing_tools
        
        return False
    
    def _has_sla_enforcement(self, node: 'NodeSpec') -> bool:
        """Check if node has SLA enforcement."""
        return (hasattr(node.data, 'sla') and 
                node.data.sla and 
                hasattr(node.data.sla, 'enforce_usage') and 
                node.data.sla.enforce_usage)
    
    def _format_sla_requirements(self, sla) -> str:
        """Format SLA requirements into readable string."""
        requirements = []
        
        if hasattr(sla, 'tool_usage_required') and sla.tool_usage_required:
            requirements.append("must use tools")
        
        if hasattr(sla, 'min_tool_calls') and sla.min_tool_calls:
            requirements.append(f"min {sla.min_tool_calls} tool calls")
        
        if hasattr(sla, 'required_tools') and sla.required_tools:
            requirements.append(f"required: {', '.join(sla.required_tools)}")
        
        if hasattr(sla, 'final_tool_must_be') and sla.final_tool_must_be:
            requirements.append(f"must end with: {sla.final_tool_must_be}")
        
        return "; ".join(requirements) if requirements else "enforce usage"
    
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
        
        # Add routing consistency validation
        routing_issues = self.validate_routing_consistency()
        issues.extend(routing_issues)
        
        # Validate node-type specific requirements
        for node in self.nodes:
            # Validate SLA configuration if present
            if node.data.sla:
                sla = node.data.sla
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
            
            if node.type == "tool":
                if not node.data.tool_name:
                    issues.append(f"Tool node '{node.id}' ({node.label}) missing required 'tool_name'")
                elif tool_catalog and node.data.tool_name not in tool_catalog:
                    issues.append(f"🚨 TOOL HALLUCINATION: Node '{node.id}' uses non-existent tool '{node.data.tool_name}'. Available tools: {sorted(tool_catalog.keys())}")
                elif tool_catalog and node.data.tool_name in tool_catalog:
                    # Validate tool parameters
                    tool_info = tool_catalog[node.data.tool_name]
                    required_params = tool_info.get("required_parameters", [])
                    tool_info.get("parameters", {})
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
        
        # Check nodes with routing tools have proper edge wiring
        routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
        
        for node in self.nodes:
            has_routing_tool = False
            
            # Check if node uses routing tools
            if node.type == "decision" and node.data.tool_name in routing_tools:
                has_routing_tool = True
            elif node.type == "agent" and node.data.tools:
                has_routing_tool = any(tool in routing_tools for tool in node.data.tools)
            
            if has_routing_tool:
                node_edges = outgoing_edges.get(node.id, [])
                
                # Must have outgoing edges
                if not node_edges:
                    issues.append(f"🚨 DANGLING ROUTING NODE: '{node.id}' ({node.label}) uses routing tools but has no outgoing edges")
                    continue
                
                # Check for conditional vs unconditional edges
                conditional_edges = [e for e in node_edges if e.data and e.data.condition]
                [e for e in node_edges if not (e.data and e.data.condition)]
                
                if not conditional_edges:
                    issues.append(f"🚨 MISSING CONDITIONS: Routing node '{node.id}' ({node.label}) has outgoing edges but none have conditions")
                
                # All targets must exist
                for edge in node_edges:
                    target_exists = any(n.id == edge.target for n in self.nodes)
                    if not target_exists:
                        issues.append(f"🚨 BROKEN EDGE: Edge '{edge.id}' points to non-existent target '{edge.target}'")
        
        # Check for orphaned conditional edges (edges with conditions from non-routing nodes)
        for edge in self.edges:
            if edge.data and edge.data.condition:
                source_node = next((n for n in self.nodes if n.id == edge.source), None)
                if source_node:
                    has_routing_tool = False
                    
                    if source_node.type == "decision" and source_node.data.tool_name in routing_tools:
                        has_routing_tool = True
                    elif source_node.type == "agent" and source_node.data.tools:
                        has_routing_tool = any(tool in routing_tools for tool in source_node.data.tools)
                    
                    if not has_routing_tool:
                        issues.append(f"⚠️ ORPHANED CONDITION: Edge '{edge.id}' has condition '{edge.data.condition}' but source node '{edge.source}' doesn't use routing tools")
        
        # Check for unreachable nodes (no incoming edges)
        all_targets = {edge.target for edge in self.edges}
        all_sources = {edge.source for edge in self.edges}
        
        for node in self.nodes:
            if node.id not in all_targets and node.id in all_sources:
                # This is a root node (has outgoing but no incoming) - OK
                continue
            elif node.id not in all_targets and node.id not in all_sources:
                # Completely isolated node
                issues.append(f"🚨 ISOLATED NODE: Node '{node.id}' ({node.label}) has no connections")
        
        # 4. SEMANTIC ROUTE VALIDATION: Check route names match between tools and edges
        for node in self.nodes:
            has_routing_tools = False
            routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
            
            # Check both agent nodes with tools and decision nodes with tool_name
            if node.type == 'agent' and node.data.tools:
                has_routing_tools = any(tool in routing_tools for tool in node.data.tools)
            elif node.type == 'decision' and node.data.tool_name in routing_tools:
                has_routing_tools = True
                
            if has_routing_tools and node.id in outgoing_edges:
                conditional_edges = [e for e in outgoing_edges[node.id] if e.data.condition]
                
                # Check for incompatible condition patterns
                for edge in conditional_edges:
                    condition = edge.data.condition or ""
                    
                    # CRITICAL: Check for decision== patterns when routing tools are used
                    if "decision ==" in condition and "routed_to ==" not in condition:
                        issues.append(f"🚨 ROUTING MISMATCH: Node '{node.id}' uses routing tools ({routing_tools}) but edge conditions use 'decision ==' pattern. DAG executor expects 'routed_to ==' pattern. Edge condition: '{condition}'")
                    
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