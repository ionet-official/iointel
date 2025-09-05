"""
Workflow Helper Utilities
========================
High-level helpers for generating and executing workflows from natural language prompts.
"""
import os
import sys
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

# Add the project root to the path for imports
if __name__ == "__main__":
    # Add the project root to sys.path for imports
    import os
    import sys
    # Get the project root (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(0, project_root)

# Use conditional imports to avoid circular import issues when running directly
try:
    from iointel.src.agent_methods.agents.workflow_agent import WorkflowPlanner
    from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
    from iointel.src.agent_methods.data_models.execution_models import (
        WorkflowExecutionResult, 
        ExecutionStatus,
        NodeExecutionResult,
        AgentExecutionResult,
        DataSourceResult
    )
    from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
    from iointel.src.utilities.tool_registry_utils import create_tool_catalog
    from iointel.src.utilities.dag_executor import DAGExecutor
    from iointel.src.utilities.graph_nodes import WorkflowState
    from iointel.src.agent_methods.data_models.datamodels import AgentParams
    from iointel.src.utilities.io_logger import get_component_logger
    from iointel.src.web.execution_feedback import ExecutionFeedbackCollector
except ImportError as e:
    if __name__ == "__main__":
        print(f"❌ Import error: {e}")
        print("This file should be run from the project root with: uv run python iointel/src/utilities/workflow_helpers.py")
        sys.exit(1)
    else:
        raise

logger = get_component_logger('workflow_helpers')


def _get_model_config() -> Dict[str, Any]:
    """Get shared model configuration."""
    from iointel.src.utilities.constants import get_model_config
    
    helper_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    return get_model_config(model=helper_model)


def _get_default_tool_catalog() -> Dict[str, Any]:
    """Get default tool catalog with standard settings."""
    # Load tools first to ensure registry is populated
    load_tools_from_env()
    # Use the same pattern as the web app
    return create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)


def _create_workflow_planner(conversation_id: str, debug: bool = False) -> WorkflowPlanner:
    """Create a configured WorkflowPlanner instance."""
    model_config = _get_model_config()
    
    return WorkflowPlanner(
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        conversation_id=conversation_id,
        debug=debug
    )


def _create_dag_executor(feedback_collector: Optional[ExecutionFeedbackCollector] = None) -> DAGExecutor:
    """Create a configured DAGExecutor instance."""
    import os
    max_concurrent = int(os.getenv("MAX_CONCURRENT_AGENTS", "3"))
    return DAGExecutor(
        feedback_collector=feedback_collector,
        max_concurrent_agents=max_concurrent
    )


def _create_initial_state(
    objective: str,
    conversation_id: str,
    user_inputs: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None
) -> WorkflowState:
    """Create initial workflow state with proper configuration."""
    return WorkflowState(
        initial_text=objective,
        conversation_id=conversation_id,
        user_inputs=user_inputs or {},
        results={},
        execution_id=execution_id or str(uuid4())
    )


def _build_node_results(final_state: WorkflowState, workflow_spec: WorkflowSpec) -> Dict[str, NodeExecutionResult]:
    """Build typed node results from final state."""
    node_results = {}
    
    # Create lookups for node types and labels from the workflow spec
    node_type_lookup = {node.id: node.type for node in workflow_spec.nodes}
    node_label_lookup = {node.id: node.label for node in workflow_spec.nodes}
    
    for node_id, result in final_state.results.items():
        # Get actual node type and label from workflow spec
        actual_node_type = node_type_lookup.get(node_id, "agent")
        node_label = node_label_lookup.get(node_id, node_id)  # Fallback to node_id if no label
        
        # If the label is still generic (like "agent_1"), generate a meaningful one
        if node_label and node_label.startswith(f"{actual_node_type}_"):
            # Find the actual node in the workflow spec to get more info
            actual_node = None
            for node in workflow_spec.nodes:
                if node.id == node_id:
                    actual_node = node
                    break
            
            if actual_node:
                if actual_node_type == "agent" and hasattr(actual_node.data, 'agent_instructions'):
                    instructions = actual_node.data.agent_instructions
                    if instructions:
                        # Extract first few words as label
                        words = instructions.split()[:3]
                        node_label = ' '.join(words).title()
                        if len(node_label) > 30:
                            node_label = node_label[:27] + "..."
                    else:
                        node_label = f"Agent {node_id.split('_')[-1]}"
                elif actual_node_type == "data_source" and hasattr(actual_node.data, 'source_name'):
                    source_name = actual_node.data.source_name
                    node_label = f"{source_name.replace('_', ' ').title()} Input"
                elif actual_node_type == "decision":
                    node_label = f"Decision Point {node_id.split('_')[-1]}"
                else:
                    node_label = f"{actual_node_type.replace('_', ' ').title()} {node_id.split('_')[-1]}"
        
        # Map to execution model node types
        if actual_node_type == "data_source":
            node_type = "data_source"
        elif actual_node_type == "decision":
            node_type = "decision"
        else:
            node_type = "agent"
        
        # The result itself could be typed or not
        if isinstance(result, (AgentExecutionResult, DataSourceResult)):
            node_result = result
        else:
            # Legacy result - just wrap it
            node_result = {"result": result}
        
        node_results[node_id] = NodeExecutionResult(
            node_id=node_id,
            node_label=node_label,
            node_type=node_type,
            status=ExecutionStatus.COMPLETED,
            result=node_result
        )
    
    return node_results


def _determine_execution_status(stats: Dict[str, Any]) -> ExecutionStatus:
    """Determine overall execution status based on statistics."""
    failed_count = stats.get("failed_nodes", 0)
    return ExecutionStatus.COMPLETED if failed_count == 0 else ExecutionStatus.PARTIAL


async def execute_workflow(
    workflow_spec: WorkflowSpec,
    user_inputs: Optional[Dict[str, Any]] = None,
    objective: Optional[str] = None,
    conversation_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    debug: bool = False
) -> WorkflowExecutionResult:
    """Execute a workflow spec and return results.
    
    This is the main execution function used by both tests and web app for consistent execution.
    
    Args:
        workflow_spec: The workflow spec to execute
        user_inputs: Optional user inputs for data source nodes
        objective: Optional objective override, defaults to workflow description
        conversation_id: Optional conversation ID, will generate if not provided
        execution_id: Optional execution ID, will generate if not provided
        feedback_collector: Optional feedback collector for web app
        debug: Enable debug logging
        
    Returns:
        WorkflowExecutionResult with typed execution results
        
    Example:
        result = await execute_workflow(
            workflow_spec=my_spec,
            user_inputs={"stock_symbol": "AAPL"}
        )
        if result.status == ExecutionStatus.COMPLETED:
            print("Success!")
    """
    conversation_id = conversation_id or str(uuid4())
    execution_id = execution_id or str(uuid4())
    start_time = datetime.now()
    
    # Use workflow description as objective if not provided
    if objective is None:
        objective = workflow_spec.description or workflow_spec.title
    
    # Create initial state
    initial_state = _create_initial_state(
        objective=objective,
        conversation_id=conversation_id,
        user_inputs=user_inputs,
        execution_id=execution_id
    )
    
    if debug:
        logger.info(f"🚀 Executing workflow: {workflow_spec.title}")
        logger.info(f"   Objective: {objective}")
        if user_inputs:
            logger.info(f"   User inputs: {list(user_inputs.keys())}")
    
    try:
        # Create DAG executor
        executor = _create_dag_executor(feedback_collector)
        
        # Build execution graph
        executor.build_execution_graph(
            workflow_spec=workflow_spec,
            objective=objective,
            conversation_id=conversation_id
        )
        
        # Validate DAG
        dag_issues = executor.validate_dag()
        if dag_issues and debug:
            logger.warning(f"DAG validation issues: {dag_issues}")
        
        # Execute workflow
        final_state = await executor.execute_dag(initial_state)
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        
        if debug:
            logger.info(f"✅ Execution completed: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
            logger.info(f"   Efficiency: {stats['execution_efficiency']}")
        
        # Build typed node results from state results
        node_results = _build_node_results(final_state, workflow_spec)
        
        # Determine overall status
        overall_status = _determine_execution_status(stats)
        
        # Extract execution summary if available from state
        execution_summary_data = final_state.execution_summary if final_state.execution_summary else None
        if execution_summary_data:
            logger.info(f"📊 Got execution summary with {len(execution_summary_data.nodes_executed)} nodes")
        
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=overall_status,
            node_results=node_results,
            final_output=final_state.results,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "stats": stats,
                "conversation_id": conversation_id,
                "execution_summary": execution_summary_data
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {e}")
        
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error=str(e),
            metadata={
                "conversation_id": conversation_id
            }
        )


async def execute_workflow_with_metadata(
    workflow_spec: WorkflowSpec,
    execution_id: str,
    user_inputs: Optional[Dict[str, Any]] = None,
    form_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    client_mode: bool = True,
    debug: bool = False
) -> WorkflowExecutionResult:
    """Execute a workflow with full metadata support, matching web app behavior.
    
    This function matches the execution pattern used by the web app's
    execute_workflow_background function, ensuring consistent behavior.
    
    Args:
        workflow_spec: The workflow spec to execute
        execution_id: Unique execution ID for tracking
        user_inputs: Optional user inputs for data source nodes
        form_id: Optional form ID for tracking
        conversation_id: Optional conversation ID, will generate if not provided
        feedback_collector: Optional feedback collector for real-time updates
        client_mode: Whether to run in client mode (default True)
        debug: Enable debug logging
        
    Returns:
        WorkflowExecutionResult with typed execution results
    """
    conversation_id = conversation_id or f"execution_{execution_id}"
    start_time = datetime.now()
    
    if debug:
        logger.info(f"🛠️ Starting execution: {execution_id}")
        if user_inputs:
            logger.info(f"📝 User inputs provided: {list(user_inputs.keys())}")
        logger.info(f"📋 Executing workflow with {len(workflow_spec.nodes)} nodes")
    
    try:
        # Create DAG executor with typed execution and feedback tracking
        executor = _create_dag_executor(feedback_collector)
        
        # Build execution graph with user inputs in metadata (matching web app)
        executor.build_execution_graph(
            workflow_spec=workflow_spec,
            objective=workflow_spec.description,
            conversation_id=conversation_id,
            execution_metadata_by_node={
                node.id: {
                    "execution_id": execution_id,
                    "user_inputs": user_inputs or {},
                    "form_id": form_id,
                    "client_mode": client_mode
                }
                for node in workflow_spec.nodes
            }
        )
        
        # Create initial state for DAG execution with execution_id
        initial_state = _create_initial_state(
            objective=workflow_spec.description,
            conversation_id=conversation_id,
            user_inputs=user_inputs,
            execution_id=execution_id
        )
        
        # Execute the DAG - execution_id flows through state
        final_state = await executor.execute_dag(initial_state)
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        
        if debug:
            logger.info(f"✅ Execution completed: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
        
        # Get execution summary from state (set by DAG executor)
        execution_summary = final_state.execution_summary if final_state.execution_summary else None
        if execution_summary:
            logger.info(f"📊 Returning execution_summary with {len(execution_summary.nodes_executed)} nodes")
        
        # Build typed node results from state results
        node_results = _build_node_results(final_state, workflow_spec)
        
        # Determine overall status
        overall_status = _determine_execution_status(stats)
        
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=overall_status,
            node_results=node_results,
            final_output=final_state.results,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "stats": stats,
                "conversation_id": conversation_id,
                "execution_summary": execution_summary,
                "form_id": form_id,
                "client_mode": client_mode
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {e}")
        
        # Complete feedback tracking with error if collector provided
        if feedback_collector:
            feedback_collector.complete_execution(
                execution_id=execution_id,
                final_outputs={},
                error_summary=str(e)
            )
        
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error=str(e),
            metadata={
                "conversation_id": conversation_id,
                "form_id": form_id,
                "client_mode": client_mode
            }
        )


async def plan_and_execute(
    prompt: str,
    initial_state: Optional[WorkflowState] = None,
    tool_catalog: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    debug: bool = False
) -> WorkflowExecutionResult:
    """
    Generate a workflow from natural language and execute it immediately.
    
    This is the high-level helper that combines:
    1. WorkflowPlanner to generate DAG from prompt
    2. DAGExecutor to run the generated workflow
    
    Args:
        prompt: Natural language description of what to do
        initial_state: Optional initial workflow state
        tool_catalog: Optional tool catalog (uses default if not provided)
        conversation_id: Optional conversation ID for tracking
        max_retries: Max retries for workflow generation
        debug: Enable debug logging
        
    Returns:
        WorkflowExecutionResult containing:
        - workflow_id: The workflow ID
        - workflow_name: The workflow name
        - status: ExecutionStatus (COMPLETED or FAILED)
        - node_results: Dict of node results
        - final_output: Final workflow state results dict
        - execution_time: Time taken to execute workflow
        - metadata: Dict containing:
            - stats: Execution statistics
            - conversation_id: The conversation ID used
            - execution_summary: Execution summary if available
        
    Example:
        result = await plan_and_execute(
            "Search for TSLA news and route to buy/sell recommendation based on sentiment"
        )
    """
    conversation_id = conversation_id or str(uuid4())
    start_time = datetime.now()
    
    # Step 1: Generate workflow from prompt
    logger.info(f"📋 Generating workflow from prompt: {prompt[:100]}...")
    
    planner = _create_workflow_planner(conversation_id, debug)
    
    if tool_catalog is None:
        tool_catalog = _get_default_tool_catalog()
    
    try:
        workflow_spec = await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries
        )
        
        logger.info(f"✅ Generated workflow: {workflow_spec.title}")
        logger.info(f"   Nodes: {len(workflow_spec.nodes)}, Edges: {len(workflow_spec.edges)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate workflow: {e}")
        return WorkflowExecutionResult(
            workflow_id="generated_workflow",
            workflow_name="Generated Workflow",
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error=f"Failed to generate workflow: {str(e)}",
            metadata={
                "conversation_id": conversation_id,
                "generation_failed": True
            }
        )
    
    # Step 2: Execute the generated workflow using the main execution function
    logger.info(f"🚀 Executing workflow: {workflow_spec.title}")
    
    try:
        # Use first user input as objective if available, otherwise use prompt
        objective = prompt
        if initial_state and initial_state.user_inputs:
            # Get first user input value as the objective
            first_input = next(iter(initial_state.user_inputs.values()))
            objective = first_input
        
        # Use the main execute_workflow function
        execution_result = await execute_workflow(
            workflow_spec=workflow_spec,
            user_inputs=initial_state.user_inputs if initial_state else None,
            objective=objective,
            conversation_id=conversation_id,
            debug=debug
        )
        
        # Add generation metadata to the result
        execution_result.metadata["generation_prompt"] = prompt
        execution_result.metadata["generation_successful"] = True
        
        return execution_result
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {e}")
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error=f"Failed to execute workflow: {str(e)}",
            metadata={
                "conversation_id": conversation_id,
                "generation_successful": True,
                "execution_failed": True,
                "generation_prompt": prompt
            }
        )


async def generate_only(
    prompt: str,
    tool_catalog: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    debug: bool = False
) -> Optional[WorkflowSpec]:
    """
    Just generate a workflow without executing it.
    Useful for testing workflow generation logic.
    
    Args:
        prompt: Natural language description
        tool_catalog: Optional tool catalog
        conversation_id: Optional conversation ID for tracking
        max_retries: Max generation retries
        debug: Enable debug logging
        
    Returns:
        WorkflowSpec or None if generation failed
    """
    conversation_id = conversation_id or str(uuid4())
    
    planner = _create_workflow_planner(conversation_id, debug)
    
    if tool_catalog is None:
        tool_catalog = _get_default_tool_catalog()
    
    try:
        return await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries
        )
    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        return None


def get_execution_summary(executor: DAGExecutor) -> Dict[str, Any]:
    """Get a detailed summary of workflow execution.
    
    Args:
        executor: DAGExecutor that has completed execution
        
    Returns:
        Dict with execution summary including:
        - Basic statistics from get_execution_statistics()
        - Execution plan from get_execution_summary()
        - Success/failure metrics
    """
    stats = executor.get_execution_statistics()
    summary = executor.get_execution_summary()
    
    # Success means no failures occurred, NOT that all nodes executed
    # Conditional DAGs intentionally skip nodes!
    failed_count = stats.get("failed_nodes", 0)
    
    return {
        **stats,
        **summary,
        "success": failed_count == 0,
        "has_failures": failed_count > 0,
        "failed_count": failed_count
    }


if __name__ == "__main__":
    import asyncio
    
    async def test_workflow_helpers():
        """Test the workflow helpers functionality."""
        print("🧪 Testing workflow helpers...")
        
        # Test plan_and_execute with a simple math query
        try:
            result = await plan_and_execute(
                prompt="create a simple calculator agent with user input. YOU MUST USE CALCULATOR TOOLS TO SOLVE THE PROBLEM",
                conversation_id=str(uuid4()),
                debug=True
            )
            
            print("✅ Test completed!")
            print(f"   Status: {result.status}")
            print(f"   Workflow: {result.workflow_name}")
            print(f"   Execution time: {result.execution_time:.2f}s")
            print(f"   Nodes executed: {len(result.node_results)}")
            
            if result.final_output:
                print(f"   Final output keys: {list(result.final_output.keys())}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Run the test
    asyncio.run(test_workflow_helpers())