"""
DAG Topology Execution Engine
=============================

This module provides proper DAG (Directed Acyclic Graph) execution that respects
edge topology, enables parallel execution of independent branches, and implements
proper dependency management.

Replaces the linear chain execution in workflow.py with true graph topology.
"""

import asyncio
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

from ..agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec
from .graph_nodes import WorkflowState, TaskNode, make_task_node
from .helpers import make_logger

logger = make_logger(__name__)


@dataclass
class DAGNode:
    """Represents a node in the DAG with its dependencies and dependents."""
    node_spec: NodeSpec
    task_node_class: type[TaskNode]
    dependencies: Set[str]  # Node IDs this node depends on
    dependents: Set[str]    # Node IDs that depend on this node
    

class DAGExecutor:
    """
    Executes workflows as proper DAGs with parallel execution of independent branches.
    Supports conditional execution gating through decision nodes.
    """
    
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.execution_order: List[List[str]] = []  # List of batches that can run in parallel
        self.edges: List[EdgeSpec] = []  # Store edges for conditional checking
        self.skipped_nodes: Set[str] = set()  # Track nodes skipped due to decision gating
    
    def build_execution_graph(
        self, 
        nodes: List[NodeSpec], 
        edges: List[EdgeSpec],
        objective: str = "",
        agents: Optional[List[Any]] = None,
        conversation_id: Optional[str] = None,
        execution_metadata_by_node: Optional[Dict[str, Dict]] = None,
        agents_by_node: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, DAGNode]:
        """
        Build a DAG from nodes and edges, with proper dependency tracking.
        
        Args:
            nodes: List of workflow nodes
            edges: List of edges defining dependencies
            objective: Workflow objective
            agents: Default agents
            conversation_id: Conversation ID for task nodes
            
        Returns:
            Dictionary mapping node IDs to DAGNode objects
        """
        logger.info(f"Building DAG with {len(nodes)} nodes and {len(edges)} edges")
        
        # Initialize nodes
        self.nodes = {}
        for node in nodes:
            # Create base task data
            task_data = {
                "task_id": node.id,
                "name": node.label,
                "type": node.type,
                "objective": f"Execute {node.label}",
                "task_metadata": {
                    "config": node.data.config,
                    "tool_name": node.data.tool_name,
                    "agent_instructions": node.data.agent_instructions,
                    "workflow_id": node.data.workflow_id,
                    "ports": {
                        "inputs": node.data.ins,
                        "outputs": node.data.outs
                    }
                }
            }
            
            # Add execution metadata if available
            if execution_metadata_by_node and node.id in execution_metadata_by_node:
                task_data["execution_metadata"] = execution_metadata_by_node[node.id]
            
            # Ensure node_id is in the execution metadata for tool context
            if "execution_metadata" not in task_data:
                task_data["execution_metadata"] = {}
            task_data["execution_metadata"]["node_id"] = node.id
            task_data["execution_metadata"]["task_id"] = node.id  # Alias for compatibility
            
            # Determine agents for this node
            node_agents = None
            if agents_by_node and node.id in agents_by_node:
                node_agents = agents_by_node[node.id]
                print(f"🔧 Using {len(node_agents)} task-specific agents for node {node.id}")
            else:
                node_agents = agents or []
                if node.type == "agent" and len(node_agents) == 0:
                    print(f"⚠️ WARNING: Agent node {node.id} has no agents!")
            
            task_node_class = make_task_node(
                task=task_data,
                default_text=objective,
                default_agents=node_agents,
                conv_id=conversation_id or "default"
            )
            
            self.nodes[node.id] = DAGNode(
                node_spec=node,
                task_node_class=task_node_class,
                dependencies=set(),
                dependents=set()
            )
        
        # Store edges for conditional checking
        self.edges = edges
        
        # Build dependency relationships from edges
        for edge in edges:
            if edge.source not in self.nodes:
                logger.warning(f"Edge source '{edge.source}' not found in nodes")
                continue
            if edge.target not in self.nodes:
                logger.warning(f"Edge target '{edge.target}' not found in nodes")
                continue
            
            # source → target means target depends on source
            self.nodes[edge.target].dependencies.add(edge.source)
            self.nodes[edge.source].dependents.add(edge.target)
        
        # Compute execution order using topological sort
        self.execution_order = self._topological_sort()
        
        logger.info(f"DAG built successfully with {len(self.execution_order)} execution batches")
        for i, batch in enumerate(self.execution_order):
            logger.info(f"  Batch {i}: {batch}")
        
        return self.nodes
    
    def _topological_sort(self) -> List[List[str]]:
        """
        Perform topological sort to determine execution order.
        Returns batches of nodes that can execute in parallel.
        """
        # Kahn's algorithm for topological sorting with batching
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        execution_batches = []
        
        while queue:
            # All nodes in current queue can execute in parallel
            current_batch = []
            batch_size = len(queue)
            
            for _ in range(batch_size):
                node_id = queue.popleft()
                current_batch.append(node_id)
                
                # Reduce in-degree of dependent nodes
                for dependent_id in self.nodes[node_id].dependents:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)
            
            execution_batches.append(current_batch)
        
        # Verify we processed all nodes (no cycles)
        total_processed = sum(len(batch) for batch in execution_batches)
        if total_processed != len(self.nodes):
            unprocessed = set(self.nodes.keys()) - {node for batch in execution_batches for node in batch}
            raise ValueError(f"Cycle detected in DAG. Unprocessed nodes: {unprocessed}")
        
        return execution_batches
    
    def _should_execute_node(self, node_id: str, state: WorkflowState) -> bool:
        """
        Check if a node should be executed based on decision node results.
        
        Args:
            node_id: Node to check
            state: Current workflow state with results
            
        Returns:
            True if node should execute, False if it should be skipped
        """
        # Check if any dependency is a decision node that has gated this node
        for dep_id in self.nodes[node_id].dependencies:
            dep_node = self.nodes[dep_id]
            
            # Check if this is a decision node OR an agent node with routing information
            if dep_node.node_spec.type == "decision" or dep_node.node_spec.type == "agent":
                dep_result = state.results.get(dep_id)
                if dep_result:
                    # Check if result contains routing information
                    routed_to = None
                    
                    # Support both dict and Pydantic model formats
                    if isinstance(dep_result, dict):
                        # First check direct routed_to field
                        routed_to = dep_result.get("routed_to")
                        
                        # If not found, check for tool_usage_results with conditional_gate
                        if not routed_to and "tool_usage_results" in dep_result:
                            # Get the LAST (most recent) conditional_gate result, not the first
                            conditional_gate_results = []
                            for tool_result in dep_result["tool_usage_results"]:
                                if hasattr(tool_result, "tool_name") and tool_result.tool_name == "conditional_gate":
                                    conditional_gate_results.append(tool_result)
                            
                            # Use the last result (most recent tool call)
                            if conditional_gate_results:
                                tool_result = conditional_gate_results[-1]
                                if hasattr(tool_result, "tool_result"):
                                    gate_result = tool_result.tool_result
                                    if hasattr(gate_result, "routed_to"):
                                        routed_to = gate_result.routed_to
                                        print(f"🔍 Found routing decision from tool (last result): {routed_to}")
                                    elif isinstance(gate_result, dict) and "routed_to" in gate_result:
                                        routed_to = gate_result["routed_to"]
                                        print(f"🔍 Found routing decision from tool dict (last result): {routed_to}")
                    else:
                        # Pydantic model (like GateResult)
                        routed_to = getattr(dep_result, "routed_to", None)
                    
                    if routed_to:
                        # Find edges from this decision node to our target node
                        target_edges = [e for e in self.edges if e.source == dep_id and e.target == node_id]
                        if target_edges:
                            # Check if any edge condition matches the routing result
                            for edge in target_edges:
                                if edge.data and edge.data.condition:
                                    # Evaluate the condition properly instead of substring matching
                                    condition = edge.data.condition
                                    
                                    # Handle common condition patterns
                                    if "routed_to ==" in condition:
                                        # Extract the expected route value from condition like "routed_to == 'sell'"
                                        import re
                                        match = re.search(r"routed_to\s*==\s*['\"]([^'\"]+)['\"]", condition)
                                        if match:
                                            expected_route = match.group(1)
                                            # Check if the actual route matches the expected route
                                            # Handle both exact matches and _path suffix variations
                                            actual_route = str(routed_to)
                                            if (actual_route == expected_route or 
                                                actual_route == expected_route + "_path" or
                                                actual_route.replace("_path", "") == expected_route):
                                                logger.info(f"  ✅ Node {node_id} matches decision route: {routed_to} → {expected_route}")
                                                print(f"🎯 Node {node_id} will execute - matches route: {routed_to} → {expected_route}")
                                                return True
                                    else:
                                        # Fallback to substring matching for other condition types
                                        if str(routed_to) in condition:
                                            logger.info(f"  ✅ Node {node_id} matches decision route: {routed_to}")
                                            print(f"🎯 Node {node_id} will execute - matches route: {routed_to}")
                                            return True
                                elif not edge.data or not edge.data.condition:
                                    # If no condition is specified, execute the node
                                    logger.info(f"  ✅ Node {node_id} has no condition, executing")
                                    return True
                            
                            # No matching edge condition found - this node should be skipped
                            logger.info(f"  ⏭️  Node {node_id} skipped - decision routed to: {routed_to}")
                            print(f"⏭️ Node {node_id} will be skipped - routed to: {routed_to}")
                            return False
                    
                    # Check for simple boolean result (for boolean_mux, etc.)
                    result_value = None
                    if isinstance(dep_result, dict):
                        result_value = dep_result.get("result")
                    else:
                        result_value = getattr(dep_result, "result", None)
                    
                    if result_value is not None and isinstance(result_value, bool):
                        if not result_value:
                            logger.info(f"  ⏭️  Node {node_id} skipped - decision result: False")
                            return False
        
        return True
    
    async def execute_dag(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Execute the DAG respecting dependencies and enabling parallel execution.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state with all results
        """
        logger.info("Starting DAG execution")
        state = initial_state
        
        for batch_idx, batch in enumerate(self.execution_order):
            logger.info(f"Executing batch {batch_idx}: {batch}")
            
            if len(batch) == 1:
                # Single node - check if it should execute
                node_id = batch[0]
                if self._should_execute_node(node_id, state):
                    result = await self._execute_node(node_id, state)
                    state.results[node_id] = result
                    logger.info(f"  ✅ {node_id} → {result}")
                else:
                    self.skipped_nodes.add(node_id)
                    state.results[node_id] = {"status": "skipped", "reason": "decision_gated"}
                    logger.info(f"  ⏭️  {node_id} → skipped")
            else:
                # Multiple nodes - check each and execute those that should run
                logger.info(f"  🔄 Checking {len(batch)} nodes for execution")
                nodes_to_execute = []
                for node_id in batch:
                    if self._should_execute_node(node_id, state):
                        nodes_to_execute.append(node_id)
                    else:
                        self.skipped_nodes.add(node_id)
                        state.results[node_id] = {"status": "skipped", "reason": "decision_gated"}
                        logger.info(f"  ⏭️  {node_id} → skipped")
                
                if nodes_to_execute:
                    logger.info(f"  🔄 Executing {len(nodes_to_execute)} nodes in parallel")
                    results = await self._execute_batch_parallel(nodes_to_execute, state)
                    
                    # Update state with all results
                    for node_id, result in results.items():
                        state.results[node_id] = result
                        logger.info(f"  ✅ {node_id} → {result}")
                else:
                    logger.info(f"  ⏭️  All nodes in batch skipped")
        
        # Log execution summary
        executed_nodes = set(state.results.keys()) - self.skipped_nodes
        logger.info(f"DAG execution completed: {len(executed_nodes)} executed, {len(self.skipped_nodes)} skipped")
        if self.skipped_nodes:
            logger.info(f"  Skipped nodes: {list(self.skipped_nodes)}")
        
        return state
    
    async def _execute_node(self, node_id: str, state: WorkflowState) -> Any:
        """Execute a single node."""
        dag_node = self.nodes[node_id]
        
        # Create task node instance with required parameters
        # The task_node_class is created by make_task_node with class attributes
        task_node = dag_node.task_node_class(
            task=dag_node.task_node_class.task,
            default_text=dag_node.task_node_class.default_text,
            default_agents=dag_node.task_node_class.default_agents,
            conversation_id=dag_node.task_node_class.conversation_id
        )
        
        # Import here to avoid circular imports
        from pydantic_graph import GraphRunContext, End
        
        # Execute the node
        context = GraphRunContext(state=state, deps={})
        result = await task_node.run(context)
        
        # Extract result value
        if isinstance(result, End):
            # If it's an End, get the result from the updated state
            return state.results.get(node_id, None)
        else:
            # This shouldn't happen in our current system, but handle it
            return result
    
    async def _execute_batch_parallel(self, batch: List[str], state: WorkflowState) -> Dict[str, Any]:
        """Execute a batch of nodes in parallel."""
        # Create tasks for parallel execution
        tasks = []
        for node_id in batch:
            task = asyncio.create_task(
                self._execute_node(node_id, state),
                name=f"execute_{node_id}"
            )
            tasks.append((node_id, task))
        
        # Wait for all tasks to complete
        results = {}
        for node_id, task in tasks:
            try:
                result = await task
                results[node_id] = result
            except Exception as e:
                logger.error(f"Node {node_id} failed: {e}")
                raise
        
        return results
    
    def validate_dag(self) -> List[str]:
        """
        Validate the DAG structure and return any issues.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check for self-loops
        for node_id, node in self.nodes.items():
            if node_id in node.dependencies:
                issues.append(f"Node '{node_id}' has self-loop")
        
        # Check for orphaned nodes (no dependencies and no dependents)
        if len(self.nodes) > 1:  # Only check if we have multiple nodes
            orphaned = []
            for node_id, node in self.nodes.items():
                if not node.dependencies and not node.dependents:
                    orphaned.append(node_id)
            
            if orphaned:
                issues.append(f"Orphaned nodes (no connections): {orphaned}")
        
        # Check for missing dependencies
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    issues.append(f"Node '{node_id}' depends on missing node '{dep_id}'")
        
        return issues
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution plan."""
        return {
            "total_nodes": len(self.nodes),
            "total_batches": len(self.execution_order),
            "parallel_batches": len([batch for batch in self.execution_order if len(batch) > 1]),
            "max_parallelism": max(len(batch) for batch in self.execution_order) if self.execution_order else 0,
            "execution_order": self.execution_order,
            "dependencies": {
                node_id: list(node.dependencies) 
                for node_id, node in self.nodes.items()
            }
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics including skipped nodes.
        
        Returns:
            Dict with execution statistics
        """
        total_nodes = len(self.nodes)
        executed_nodes = total_nodes - len(self.skipped_nodes)
        
        return {
            "total_nodes": total_nodes,
            "executed_nodes": executed_nodes,
            "skipped_nodes": len(self.skipped_nodes),
            "skipped_node_ids": list(self.skipped_nodes),
            "execution_efficiency": f"{executed_nodes}/{total_nodes} ({100*executed_nodes/total_nodes:.1f}%)" if total_nodes > 0 else "0/0 (0%)"
        }


def create_dag_executor_from_spec(
    spec: WorkflowSpec, 
    objective: str = "",
    agents: Optional[List[Any]] = None,
    conversation_id: Optional[str] = None
) -> DAGExecutor:
    """
    Create a DAG executor from a WorkflowSpec.
    
    Args:
        spec: Workflow specification
        objective: Workflow objective
        agents: Default agents
        conversation_id: Conversation ID
        
    Returns:
        Configured DAG executor
    """
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=spec.nodes,
        edges=spec.edges,
        objective=objective or spec.description,
        agents=agents,
        conversation_id=conversation_id
    )
    return executor