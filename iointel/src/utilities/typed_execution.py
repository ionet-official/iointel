"""Typed execution interfaces to replace dict-based task system."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from ..agent_methods.data_models.workflow_spec import NodeSpec, WorkflowSpec, EdgeSpec
from ..agents import Agent
from .graph_nodes import WorkflowState
from ..utilities.io_logger import get_component_logger

logger = get_component_logger("TYPED_EXECUTION")


@dataclass
class TypedExecutionContext:
    """Execution context with full workflow awareness."""
    workflow_spec: WorkflowSpec      # The FUNDAMENTAL object - complete workflow
    current_node_id: str            # Which node we're executing
    state: WorkflowState            # Accumulated results/state
    agents: Optional[List[Agent]] = None  # Available agents
    conversation_id: Optional[str] = None
    objective: Optional[str] = None  # Legacy support
    execution_metadata: Optional[Dict[str, Any]] = None  # Runtime metadata (user_inputs, etc.)
    
    @property
    def node(self) -> NodeSpec:
        """Get the current node being executed (legacy compatibility)."""
        return self.current_node
    
    @property
    def current_node(self) -> NodeSpec:
        """Get the current node being executed."""
        for node in self.workflow_spec.nodes:
            if node.id == self.current_node_id:
                return node
        raise ValueError(f"Node {self.current_node_id} not found in workflow")
    
    @property
    def node_id(self) -> str:
        """Legacy compatibility."""
        return self.current_node_id
    
    @property
    def node_type(self) -> str:
        return self.current_node.type
    
    @property
    def node_data(self):
        """Access node data directly."""
        return self.current_node.data
    
    @property
    def agent_instructions(self) -> Optional[str]:
        """Get agent instructions if this is an agent node."""
        return getattr(self.node_data, 'agent_instructions', None)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get node configuration."""
        return getattr(self.node_data, 'config', None) or {}
    
    @property
    def available_results(self) -> Dict[str, Any]:
        """Get all available results from previous nodes."""
        return self.state.results
    
    @property
    def incoming_edges(self) -> List[EdgeSpec]:
        """Get edges that target this node."""
        return [e for e in self.workflow_spec.edges if e.target == self.current_node_id]
    
    @property
    def outgoing_edges(self) -> List[EdgeSpec]:
        """Get edges that originate from this node."""
        return [e for e in self.workflow_spec.edges if e.source == self.current_node_id]
    
    def get_dependencies(self) -> List[str]:
        """Get IDs of all nodes this node depends on."""
        return [edge.source for edge in self.incoming_edges]
    
    def get_input_data(self) -> Dict[str, Any]:
        """
        Get all input data for this node based on incoming edges.
        Maps source outputs to this node's expected inputs.
        """
        input_data = {}
        
        for edge in self.incoming_edges:
            source_result = self.state.results.get(edge.source)
            if source_result:
                # Handle different result formats
                if isinstance(source_result, dict) and 'result' in source_result:
                    # Extract the actual result value
                    value = source_result['result']
                elif hasattr(source_result, 'result'):
                    # Handle Pydantic models
                    value = source_result.result
                else:
                    value = source_result
                
                # Map based on edge handles if specified
                if edge.targetHandle:
                    input_data[edge.targetHandle] = value
                else:
                    # Fallback: use source node ID as key
                    input_data[edge.source] = value
                    
        return input_data
    
    def get_input_value(self, input_name: str) -> Any:
        """
        Get the value for a specific input port.
        
        This method looks through the state results to find values
        that should be connected to this node's input ports.
        """
        # Look for results that match the expected pattern
        # For example, if input_name is "user_message", look for
        # results from nodes that output to this input
        
        # This is where we'd use edge information to properly
        # map outputs to inputs based on the workflow structure
        
        # For now, return None - this needs proper implementation
        # based on edge connections
        logger.warning(f"get_input_value not fully implemented for {input_name}")
        return None
    
    def resolve_variables(self, text: str) -> str:
        """
        Resolve {node_id.field} variables in text using DataFlowResolver.
        
        Args:
            text: Text containing variable references
            
        Returns:
            Text with variables resolved
        """
        if not text or not isinstance(text, str):
            return text
            
        from .data_flow_resolver import data_flow_resolver
        return data_flow_resolver._resolve_value(text, self.available_results)
    
    def resolve_config(self) -> Dict[str, Any]:
        """
        Resolve all variables in the node's config.
        
        Returns:
            Config with all {node_id.field} references resolved
        """
        if not self.config:
            return {}
            
        from .data_flow_resolver import data_flow_resolver
        return data_flow_resolver.resolve_config(self.config, self.available_results)


class TypedNodeExecutor:
    """Base class for typed node execution."""
    
    @staticmethod
    async def execute(context: TypedExecutionContext) -> Any:
        """Execute a node based on its type."""
        logger.info(f"Executing node {context.node_id} (type: {context.node_type})")
        
        if context.node_type == "agent":
            from .typed_executors import execute_agent_typed
            return await execute_agent_typed(context)
        elif context.node_type == "data_source":
            from .typed_executors import execute_data_source_typed
            return await execute_data_source_typed(context)
        elif context.node_type == "decision":
            from .typed_executors import execute_decision_typed
            return await execute_decision_typed(context)
        elif context.node_type == "tool":
            from .typed_executors import execute_tool_typed
            return await execute_tool_typed(context)
        else:
            raise ValueError(f"Unknown node type: {context.node_type}")


# Bridge function to convert from old dict-based system to typed
def create_typed_context_from_task(
    task: dict,
    node: NodeSpec,
    state: WorkflowState,
    agents: Optional[List[Agent]] = None,
    conversation_id: Optional[str] = None
) -> TypedExecutionContext:
    """
    Create a typed execution context from the old dict-based task.
    
    This is a bridge function for gradual migration.
    """
    return TypedExecutionContext(
        node=node,
        state=state,
        objective=task.get("objective"),
        agents=agents or task.get("agents", []),
        conversation_id=conversation_id or task.get("conversation_id")
    )