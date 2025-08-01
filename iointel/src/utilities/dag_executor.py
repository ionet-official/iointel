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
from collections import deque
from dataclasses import dataclass

from ..agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec
from .graph_nodes import WorkflowState, TaskNode, make_task_node
from .node_execution_wrapper import node_execution_wrapper
from .io_logger import get_component_logger

logger = get_component_logger("DAG_EXECUTOR")


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
    
    def __init__(self, use_typed_execution: bool = False):
        self.nodes: Dict[str, DAGNode] = {}
        self.execution_order: List[List[str]] = []  # List of batches that can run in parallel
        self.edges: List[EdgeSpec] = []  # Store edges for conditional checking
        self.skipped_nodes: Set[str] = set()  # Track nodes skipped due to decision gating
        self.use_typed_execution = use_typed_execution
        self.conversation_id: Optional[str] = None
        self.agents: Optional[List[Any]] = None
        self.workflow_spec: Optional[WorkflowSpec] = None  # Store the full workflow spec
    
    def build_execution_graph(
        self, 
        workflow_spec: WorkflowSpec,
        objective: str = "",
        agents: Optional[List[Any]] = None,
        conversation_id: Optional[str] = None,
        execution_metadata_by_node: Optional[Dict[str, Dict]] = None,
        agents_by_node: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, DAGNode]:
        """
        Build a DAG from workflow spec with proper dependency tracking.
        
        Args:
            workflow_spec: Complete workflow specification
            objective: Workflow objective
            agents: Default agents
            conversation_id: Conversation ID for task nodes
            
        Returns:
            Dictionary mapping node IDs to DAGNode objects
        """
        # Store the workflow spec
        self.workflow_spec = workflow_spec
        nodes = workflow_spec.nodes
        edges = workflow_spec.edges
        
        logger.info(f"Building DAG with {len(nodes)} nodes and {len(edges)} edges")
        
        # Initialize nodes
        self.nodes = {}
        for node in nodes:
            # Create base task data
            # For agent nodes with inputs, the objective will be dynamically resolved from input data
            # This allows agents to receive the actual user input instead of a generic "Execute X" message
            node_objective = f"Execute {node.label}"
            
            # Special handling for user_input nodes - extract the actual input value
            if node.type == "data_source" and node.data.source_name == "user_input":
                # Extract user input from node config
                config = node.data.config or {}
                if "message" in config:
                    node_objective = config["message"]
                elif "default_value" in config:
                    node_objective = config["default_value"]
                logger.info(f"📝 User input node '{node.id}' has value: {node_objective}")
            
            # For agent nodes that depend on user input, we'll resolve this dynamically
            elif node.type == "agent" and node.data.ins:
                # Check if this agent depends on a user_input node
                for edge in edges:
                    if edge.target == node.id:
                        source_node = next((n for n in nodes if n.id == edge.source), None)
                        if source_node and source_node.type == "data_source" and source_node.data.source_name == "user_input":
                            # This agent depends on user input - get the value
                            config = source_node.data.config or {}
                            if "message" in config:
                                node_objective = config["message"]
                            elif "default_value" in config:
                                node_objective = config["default_value"]
                            else:
                                node_objective = ""  # Empty if no input provided
                            logger.info(f"🎯 Agent node '{node.id}' will receive user input as objective: {node_objective}")
                            break 
                
            task_data = {
                "task_id": node.id,
                "name": node.label,
                "type": node.type,
                "objective": node_objective,
                "task_metadata": {
                    "config": node.data.config,
                    "tool_name": getattr(node.data, 'source_name', None) or getattr(node.data, 'tool_name', None),
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
                if (node.type in ["agent", "decision"]) and len(node_agents) == 0:
                    # Auto-create agents from WorkflowSpec node data
                    # Decision nodes are also agents (with routing tools)
                    hydrated_agents = self._hydrate_agents_from_node(node)
                    if hydrated_agents:
                        node_agents = hydrated_agents
                        print(f"🔧 Auto-created {len(node_agents)} agents for {node.type} node {node.id}")
                    else:
                        print(f"⚠️ WARNING: {node.type} node {node.id} has no agents and no agent_instructions!")
            
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
        self.conversation_id = conversation_id
        self.agents = agents
        
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
        Check if a node should be executed based on execution_mode and dependency status.
        
        Args:
            node_id: Node to check
            state: Current workflow state with results
            
        Returns:
            True if node should execute, False if it should be skipped
        """
        node = self.nodes[node_id]
        execution_mode = getattr(node.node_spec.data, 'execution_mode', 'consolidate')
        
        # Categorize dependencies by completion status
        completed_dependencies = []
        skipped_dependencies = []
        pending_dependencies = []
        
        for dep_id in node.dependencies:
            if dep_id in self.skipped_nodes:
                skipped_dependencies.append(dep_id)
            elif dep_id in state.results:
                dep_result = state.results[dep_id]
                if isinstance(dep_result, dict) and dep_result.get("status") == "skipped":
                    skipped_dependencies.append(dep_id)
                else:
                    completed_dependencies.append(dep_id)
            else:
                pending_dependencies.append(dep_id)
        
        # Before applying execution mode logic, check for decision-based routing
        # This handles cases where routing logic determines execution
        routing_result = self._check_decision_gating(node_id, state)
        if routing_result is not None:  # None means no routing logic applies
            return routing_result
        
        if execution_mode == "consolidate":
            # Wait for ALL dependencies - traditional behavior
            if skipped_dependencies and not completed_dependencies:
                # All dependencies were skipped
                logger.info(f"  ⏭️  Node {node_id} skipped - all dependencies skipped in consolidate mode")
                return False
            elif pending_dependencies:
                # Still waiting for some dependencies
                return False
            # All non-skipped dependencies are ready
            return True
            
        elif execution_mode == "for_each":
            # Execute if we have any completed dependencies (don't wait for skipped ones)
            if completed_dependencies:
                logger.info(f"  ✅ Node {node_id} executing in for_each mode - completed: {completed_dependencies}, skipped: {skipped_dependencies}")
                return True
            elif not pending_dependencies:
                # No pending, no completed - all were skipped
                logger.info(f"  ⏭️  Node {node_id} skipped - no completed dependencies in for_each mode")
                return False
            # Still waiting for some dependencies
            return False
        
        # Fallback to consolidate behavior
        return True
    
    def _check_decision_gating(self, node_id: str, state: WorkflowState) -> Optional[bool]:
        """
        Check if any dependency is a decision node that has gated this node.
        This is now only used for decision-based routing logic.
        
        Returns:
            True if node should execute based on routing
            False if node should be skipped based on routing  
            None if no routing logic applies
        """
        for dep_id in self.nodes[node_id].dependencies:
            dep_node = self.nodes[dep_id]
            
            # Check if this is a decision node OR an agent node with routing information
            if dep_node.node_spec.type == "decision" or dep_node.node_spec.type == "agent":
                dep_result = state.results.get(dep_id)
                if dep_result:
                    # Check if result contains routing information - NEW: Extract route_index
                    route_index = None
                    routed_to = None  # Keep for backward compatibility and logging
                    
                    # Support both dict and Pydantic model formats
                    tool_usage_results = None
                    
                    # Extract tool_usage_results from different result formats
                    if hasattr(dep_result, 'agent_response') and dep_result.agent_response:
                        # AgentExecutionResult with agent_response
                        if hasattr(dep_result.agent_response, 'tool_usage_results'):
                            tool_usage_results = dep_result.agent_response.tool_usage_results
                    elif isinstance(dep_result, dict):
                        # Dict format - could be legacy or have agent_response
                        if "agent_response" in dep_result and dep_result["agent_response"]:
                            agent_resp = dep_result["agent_response"]
                            if isinstance(agent_resp, dict) and "tool_usage_results" in agent_resp:
                                tool_usage_results = agent_resp["tool_usage_results"]
                            elif hasattr(agent_resp, 'tool_usage_results'):
                                tool_usage_results = agent_resp.tool_usage_results
                        elif "tool_usage_results" in dep_result:
                            # Direct tool_usage_results in dict
                            tool_usage_results = dep_result["tool_usage_results"]
                    
                    # Look for route_index in tool_usage_results
                    if tool_usage_results:
                        # Get the LAST (most recent) conditional_gate result
                        conditional_gate_results = []
                        for tool_result in tool_usage_results:
                            tool_name = None
                            if hasattr(tool_result, "tool_name"):
                                tool_name = tool_result.tool_name
                            elif isinstance(tool_result, dict) and "tool_name" in tool_result:
                                tool_name = tool_result["tool_name"]
                            
                            if tool_name == "conditional_gate":
                                conditional_gate_results.append(tool_result)
                        
                        # Use the last result (most recent tool call)
                        if conditional_gate_results:
                            tool_result = conditional_gate_results[-1]
                            gate_result = None
                            
                            if hasattr(tool_result, "tool_result"):
                                gate_result = tool_result.tool_result
                            elif isinstance(tool_result, dict) and "tool_result" in tool_result:
                                gate_result = tool_result["tool_result"]
                            
                            if gate_result:
                                if hasattr(gate_result, "route_index"):
                                    route_index = gate_result.route_index
                                    routed_to = getattr(gate_result, "routed_to", "unknown")
                                    logger.info(f"🔍 Found routing decision from conditional_gate", data={
                                        "decision_node": dep_id,
                                        "route_index": route_index,
                                        "routed_to": routed_to,
                                        "confidence": getattr(gate_result, "confidence", 1.0),
                                        "decision_reason": getattr(gate_result, "decision_reason", ""),
                                        "result_type": "gate_result_object"
                                    })
                                elif isinstance(gate_result, dict) and "route_index" in gate_result:
                                    route_index = gate_result["route_index"]
                                    routed_to = gate_result.get("routed_to", "unknown")
                                    logger.info(f"🔍 Found routing decision from conditional_gate dict", data={
                                        "decision_node": dep_id,
                                        "route_index": route_index,
                                        "routed_to": routed_to,
                                        "result_type": "dict_format"
                                    })
                    
                    # Direct route_index check (legacy)
                    if route_index is None:
                        if hasattr(dep_result, "route_index"):
                            route_index = dep_result.route_index
                            routed_to = getattr(dep_result, "routed_to", None)
                        elif isinstance(dep_result, dict) and "route_index" in dep_result:
                            route_index = dep_result["route_index"]
                            routed_to = dep_result.get("routed_to")
                    
                    if route_index is not None:
                        # Find edges from this decision node to our target node
                        target_edges = [e for e in self.edges if e.source == dep_id and e.target == node_id]
                        if target_edges:
                            # NEW: Simple route_index matching - much cleaner!
                            for edge in target_edges:
                                if edge.data and edge.data.route_index is not None:
                                    # Direct integer comparison - no regex needed!
                                    if edge.data.route_index == route_index:
                                        logger.success(f"🎯 Node will execute - route index match", data={
                                            "node_id": node_id,
                                            "route_index": route_index,
                                            "routed_to": routed_to,
                                            "decision_node": dep_id,
                                            "edge_route_index": edge.data.route_index,
                                            "edge_route_label": edge.data.route_label,
                                            "execution_decision": "EXECUTE"
                                        })
                                        return True
                                
                                # Legacy fallback: condition-based matching for backward compatibility
                                elif edge.data and edge.data.condition and routed_to:
                                    condition = edge.data.condition
                                    if "routed_to ==" in condition:
                                        import re
                                        match = re.search(r"routed_to\s*==\s*['\"]([^'\"]+)['\"]", condition)
                                        if match:
                                            expected_route = match.group(1)
                                            actual_route = str(routed_to)
                                            if (actual_route == expected_route or 
                                                actual_route == expected_route + "_path" or
                                                actual_route.replace("_path", "") == expected_route):
                                                logger.info(f"🎯 Node will execute - legacy condition match", data={
                                                    "node_id": node_id,
                                                    "actual_route": actual_route,
                                                    "expected_route": expected_route,
                                                    "decision_node": dep_id,
                                                    "edge_condition": condition,
                                                    "execution_decision": "EXECUTE",
                                                    "match_type": "LEGACY"
                                                })
                                                return True
                                
                                # If no routing data is specified, execute the node
                                elif not edge.data or (not edge.data.route_index and not edge.data.condition):
                                    logger.info(f"  ✅ Node {node_id} has no routing data, executing")
                                    return True
                            
                            # No matching edge found - this node should be skipped
                            logger.warning(f"⏭️ Node will be skipped - no matching route", data={
                                "node_id": node_id,
                                "route_index": route_index,
                                "routed_to": routed_to,
                                "decision_node": dep_id,
                                "available_edges": len(target_edges),
                                "execution_decision": "SKIP"
                            })
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
        
        # No routing logic applies
        return None
    
    async def execute_dag(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Execute the DAG respecting dependencies and enabling parallel execution.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state with all results
        """
        logger.info("Starting DAG execution")
        logger.info(f"🎯 Execution Plan: {len(self.execution_order)} batches, max parallelism: {max(len(batch) for batch in self.execution_order)}")
        for i, batch in enumerate(self.execution_order):
            if len(batch) > 1:
                logger.info(f"  📦 Batch {i}: {batch} (parallel)")
            else:
                logger.info(f"  📦 Batch {i}: {batch} (sequential)")
        
        state = initial_state
        self._last_state = state  # Ensure _is_decision_gated_skip always has access to latest state
        
        for batch_idx, batch in enumerate(self.execution_order):
            logger.info(f"Executing batch {batch_idx}: {batch}")
            
            if len(batch) == 1:
                # Single node - check if it should execute
                node_id = batch[0]
                logger.info(f"  🔍 Evaluating single node: {node_id}")
                should_execute = self._should_execute_node(node_id, state)
                logger.info(f"  🎯 Decision for {node_id}: {'EXECUTE' if should_execute else 'SKIP'}")
                
                if should_execute:
                    logger.info(f"  🚀 Executing {node_id}...")
                    result = await self._execute_node(node_id, state)
                    state.results[node_id] = result
                    logger.info(f"  ✅ {node_id} → COMPLETED")
                else:
                    self.skipped_nodes.add(node_id)
                    state.results[node_id] = {"status": "skipped", "reason": "decision_gated"}
                    logger.info(f"  ⏭️  {node_id} → SKIPPED (decision gated)")
            else:
                # Multiple nodes - check each and execute those that should run
                logger.info(f"  🔄 Evaluating {len(batch)} nodes for parallel execution")
                nodes_to_execute = []
                for node_id in batch:
                    should_execute = self._should_execute_node(node_id, state)
                    logger.info(f"  🎯 Decision for {node_id}: {'EXECUTE' if should_execute else 'SKIP'}")
                    
                    if should_execute:
                        nodes_to_execute.append(node_id)
                    else:
                        self.skipped_nodes.add(node_id)
                        state.results[node_id] = {"status": "skipped", "reason": "decision_gated"}
                        logger.info(f"  ⏭️  {node_id} → SKIPPED (decision gated)")
                
                if nodes_to_execute:
                    logger.info(f"  🚀 Launching {len(nodes_to_execute)} nodes in parallel: {nodes_to_execute}")
                    results = await self._execute_batch_parallel(nodes_to_execute, state)
                    
                    # Update state with all results
                    for node_id, result in results.items():
                        state.results[node_id] = result
                        logger.info(f"  ✅ {node_id} → COMPLETED")
                else:
                    logger.info("  ⏭️  All nodes in batch skipped")
        
        # Log execution summary
        executed_nodes = set(state.results.keys()) - self.skipped_nodes
        logger.info(f"DAG execution completed: {len(executed_nodes)} executed, {len(self.skipped_nodes)} skipped")
        if self.skipped_nodes:
            logger.info(f"  Skipped nodes: {list(self.skipped_nodes)}")
        
        return state
    
    async def _execute_node(self, node_id: str, state: WorkflowState) -> Any:
        """Execute a single node with SLA enforcement wrapper."""
        dag_node = self.nodes[node_id]
        
        # Use typed execution if enabled
        if self.use_typed_execution:
            return await self._execute_node_typed(node_id, state)
        
        # Extract node data for SLA requirements
        node_data = dag_node.node_spec.data.model_dump() if hasattr(dag_node.node_spec.data, 'model_dump') else {}
        
        # Define the actual node execution function
        async def execute_node_core():
            # Create task node instance with required parameters
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
                return state.results.get(node_id, None)
            else:
                return result
        
        # Wrap execution with SLA enforcement using authoritative NodeSpec
        try:
            result = await node_execution_wrapper.execute_with_sla_enforcement(
                node_executor=execute_node_core,
                node_spec=dag_node.node_spec,  # Pass full NodeSpec as authoritative source
                input_data=state.results,  # Available data from previous nodes
                node_id=node_id,
                node_type=dag_node.node_spec.type,
                node_label=dag_node.node_spec.label
            )
            return result
        except Exception as e:
            logger.error(f"Node {node_id} execution failed with SLA wrapper: {e}")
            # Fall back to direct execution for compatibility
            return await execute_node_core()
    
    async def _execute_node_typed(self, node_id: str, state: WorkflowState) -> Any:
        """Execute a single node using typed execution system."""
        from .typed_execution import TypedExecutionContext, TypedNodeExecutor
        from ..utilities.io_logger import get_component_logger
        
        logger = get_component_logger("DAG_EXECUTOR")
        dag_node = self.nodes[node_id]
        
        logger.info(f"🎯 Typed execution of node {node_id}")
        
        # Create typed execution context with full workflow awareness
        context = TypedExecutionContext(
            workflow_spec=self.workflow_spec,
            current_node_id=node_id,
            state=state,
            agents=dag_node.task_node_class.default_agents,
            conversation_id=self.conversation_id,
            objective=dag_node.task_node_class.task.get("objective")  # Legacy support
        )
        
        # Execute using typed executor
        try:
            result = await TypedNodeExecutor.execute(context)
            logger.info(f"✅ Typed execution completed for {node_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Typed execution failed for {node_id}: {e}")
            raise
    
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
    
    def _hydrate_agents_from_node(self, node: NodeSpec) -> Optional[List[Any]]:
        """
        Create AgentParams from WorkflowSpec node data.
        
        Args:
            node: Node specification with agent_instructions
            
        Returns:
            List of AgentParams if successful, None otherwise
        """
        if node.type not in ["agent", "decision"] or not node.data.agent_instructions:
            return None
            
        try:
            # Import necessary modules
            from ..agent_methods.data_models.datamodels import AgentParams
            from ..utilities.registries import TOOLS_REGISTRY
            from ..utilities.constants import get_model_config
            
            # Load tools for the agent if specified
            agent_tools = []
            if node.data.tools:
                for tool_name in node.data.tools:
                    if tool_name in TOOLS_REGISTRY:
                        # Just pass the tool name - resolution happens in create_agent
                        agent_tools.append(tool_name)
                        logger.info(f"Loading tool '{tool_name}' for agent '{node.id}'")
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in registry for agent '{node.id}'")
            
            # Use centralized model configuration
            config = get_model_config(
                model=node.data.model,
                api_key=None,  # Let it use defaults
                base_url=None  # Let it use defaults
            )
            model = config["model"]
            api_key = config["api_key"]
            base_url = config["base_url"]
            
            agent_params = AgentParams(
                name=f"agent_{node.id}",
                instructions=node.data.agent_instructions,
                model=model,
                api_key=api_key,
                base_url=base_url,
                tools=agent_tools,
                context=node.data.config.get("context") if node.data.config else None,
                persona=node.data.config.get("persona") if node.data.config else None,
                model_settings=node.data.config.get("model_settings") if node.data.config else None,
            )
            return [agent_params]
            
        except Exception as e:
            logger.error(f"Failed to hydrate agents for node {node.id}: {e}")
            return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics including skipped nodes.
        Returns:
            Dict with execution statistics
        """
        total_nodes = len(self.nodes)
        # Count nodes skipped due to decision gating
        decision_gated_skips = [nid for nid in self.skipped_nodes if self._is_decision_gated_skip(nid)]
        num_decision_gated_skips = len(decision_gated_skips)
        effective_total_nodes = total_nodes - num_decision_gated_skips
        executed_nodes = total_nodes - len(self.skipped_nodes)
        # Only count as inefficient those skipped for other reasons
        efficiency = (executed_nodes / effective_total_nodes) if effective_total_nodes > 0 else 1.0
        return {
            "total_nodes": total_nodes,
            "executed_nodes": executed_nodes,
            "skipped_nodes": len(self.skipped_nodes),
            "skipped_node_ids": list(self.skipped_nodes),
            "decision_gated_skips": decision_gated_skips,
            "execution_efficiency": f"{executed_nodes}/{effective_total_nodes} ({100*efficiency:.1f}%)" if effective_total_nodes > 0 else "N/A (all skips decision-gated)"
        }

    def _is_decision_gated_skip(self, node_id: str) -> bool:
        # Returns True if the node was skipped due to decision gating
        # Check the state.results for this node if available
        # For now, check if the node's result has status 'skipped' and reason 'decision_gated'
        # This method may need to be passed state if not available as attribute
        # We'll assume state is available as self._last_state (set in execute_dag)
        if hasattr(self, '_last_state') and self._last_state:
            result = self._last_state.results.get(node_id)
            if isinstance(result, dict):
                return result.get("status") == "skipped" and result.get("reason") == "decision_gated"
        return False


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
        workflow_spec=spec,
        objective=objective or spec.description,
        agents=agents,
        conversation_id=conversation_id
    )
    return executor