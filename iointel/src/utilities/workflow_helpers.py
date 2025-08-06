"""
Workflow Helper Utilities
========================
High-level helpers for generating and executing workflows from natural language prompts.
"""
import os
from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4
from datetime import datetime

from ..agent_methods.agents.workflow_planner import WorkflowPlanner
from ..agent_methods.data_models.workflow_spec import WorkflowSpec
from ..agent_methods.data_models.execution_models import (
    WorkflowExecutionResult, 
    ExecutionStatus,
    NodeExecutionResult,
    AgentExecutionResult,
    DataSourceResult
)
from ..agent_methods.tools.tool_loader import load_tools_from_env
from ..utilities.tool_registry_utils import create_tool_catalog
from ..utilities.dag_executor import DAGExecutor
from ..utilities.graph_nodes import WorkflowState
from ..agent_methods.data_models.datamodels import AgentParams
from ..utilities.io_logger import get_component_logger
from ..web.execution_feedback import ExecutionFeedbackCollector

logger = get_component_logger('workflow_helpers')


async def plan_and_execute(
    prompt: str,
    initial_state: Optional[WorkflowState] = None,
    tool_catalog: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    debug: bool = False
) -> Dict[str, Any]:
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
        Dict containing:
        - workflow_spec: The generated WorkflowSpec
        - execution_result: Final WorkflowState after execution
        - execution_stats: Execution statistics
        - success: Boolean indicating overall success
        
    Example:
        result = await plan_and_execute(
            "Search for TSLA news and route to buy/sell recommendation based on sentiment"
        )
    """
    conversation_id = conversation_id or str(uuid4())
    
    # Step 1: Generate workflow from prompt
    logger.info(f"📋 Generating workflow from prompt: {prompt[:100]}...")
    
    # Use shared model configuration
    from .constants import get_model_config
    
    helper_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    model_config = get_model_config(model=helper_model)
    
    planner = WorkflowPlanner(
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        conversation_id=conversation_id,
        debug=debug
    )
    
    if tool_catalog is None:
        # Load tools first to ensure registry is populated
        load_tools_from_env()
        # Use the same pattern as the web app
        tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
    
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
        return {
            "workflow_spec": None,
            "execution_result": None,
            "execution_stats": None,
            "success": False,
            "error": str(e)
        }
    
    # Step 2: Execute the generated workflow
    logger.info(f"🚀 Executing workflow: {workflow_spec.title}")
    
    try:
        # Create DAG executor with typed execution
        executor = DAGExecutor(use_typed_execution=True)
        
        # Use first user input as objective if available, otherwise use prompt
        objective = prompt
        if initial_state and initial_state.user_inputs:
            # Get first user input value as the objective
            first_input = next(iter(initial_state.user_inputs.values()))
            objective = first_input
            
        executor.build_execution_graph(
            workflow_spec=workflow_spec,
            objective=objective,
            conversation_id=conversation_id
        )
        
        # Validate the DAG
        issues = executor.validate_dag()
        if issues:
            logger.warning(f"⚠️  DAG validation issues: {issues}")
        
        # Create initial state if not provided
        if initial_state is None:
            initial_state = WorkflowState(
                initial_text=prompt,
                conversation_id=conversation_id,
                results={}
            )
        
        # Execute the DAG
        final_state = await executor.execute_dag(initial_state)
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        
        logger.info("✅ Workflow execution complete!")
        logger.info(f"   Executed: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
        logger.info(f"   Efficiency: {stats['execution_efficiency']}")
        
        return {
            "workflow_spec": workflow_spec,
            "execution_result": final_state,
            "execution_stats": stats,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {e}")
        return {
            "workflow_spec": workflow_spec,
            "execution_result": None,
            "execution_stats": None,
            "success": False,
            "error": str(e)
        }


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
    
    # Use shared model configuration
    from .constants import get_model_config
    
    helper_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    model_config = get_model_config(model=helper_model)
    
    planner = WorkflowPlanner(
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        conversation_id=conversation_id, 
        debug=debug
    )
    
    if tool_catalog is None:
        # Load tools first to ensure registry is populated
        load_tools_from_env()
        # Use the same pattern as the web app
        tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
    
    try:
        return await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries
        )
    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        return None


# def analyze_workflow_spec(spec: WorkflowSpec) -> Dict[str, Any]:
#     """
#     Analyze a workflow spec for key features like execution modes, SLA, etc.
    
#     Args:
#         spec: WorkflowSpec to analyze
        
#     Returns:
#         Dict with analysis results
#     """
#     analysis = {
#         "total_nodes": len(spec.nodes),
#         "total_edges": len(spec.edges),
#         "node_types": {},
#         "execution_modes": {"consolidate": [], "for_each": []},
#         "sla_nodes": [],
#         "agents_with_tools": [],
#         "conditional_edges": [],
#         "decision_gates": []
#     }
    
#     for node in spec.nodes:
#         # Count node types
#         node_type = node.type
#         analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1
        
#         # Check execution mode
#         if hasattr(node.data, 'execution_mode'):
#             mode = node.data.execution_mode
#             analysis["execution_modes"][mode].append(node.label)
        
#         # Check for SLA
#         if hasattr(node, 'sla') and node.sla:
#             analysis["sla_nodes"].append({
#                 "node": node.label,
#                 "enforce_usage": getattr(node.sla, 'enforce_usage', False),
#                 "required_tools": getattr(node.sla, 'required_tools', []),
#                 "final_tool_must_be": getattr(node.sla, 'final_tool_must_be', None),
#                 "min_tool_calls": getattr(node.sla, 'min_tool_calls', 0)
#             })
        
#         # Check for agents with tools
#         if node.type == "agent" and hasattr(node.data, 'tools') and node.data.tools:
#             analysis["agents_with_tools"].append({
#                 "node": node.label,
#                 "tools": node.data.tools
#             })
            
#             # Check for conditional_gate
#             if "conditional_gate" in node.data.tools:
#                 analysis["decision_gates"].append(node.label)
    
#     # Check for conditional edges
#     for edge in spec.edges:
#         if edge.data and hasattr(edge.data, 'condition') and edge.data.condition:
#             analysis["conditional_edges"].append({
#                 "source": edge.source,
#                 "target": edge.target,
#                 "condition": edge.data.condition
#             })
    
#     return analysis


# # Convenience function for quick testing
# async def quick_test(prompt: str) -> None:
#     """
#     Quick test function that generates, executes, and prints results.
    
#     Args:
#         prompt: Natural language workflow description
#     """
#     print(f"\n🧪 Testing: {prompt}")
#     print("=" * 60)
    
#     result = await plan_and_execute(prompt, debug=True)
    
#     if result["success"]:
#         spec = result["workflow_spec"]
#         stats = result["execution_stats"]
        
#         print("\n✅ SUCCESS!")
#         print(f"📋 Workflow: {spec.title}")
#         print(f"📊 Execution: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
        
#         # Analyze the workflow
#         analysis = analyze_workflow_spec(spec)
#         print("\n🔍 Analysis:")
#         print(f"  - Node types: {analysis['node_types']}")
#         print(f"  - For-each nodes: {analysis['execution_modes']['for_each']}")
#         print(f"  - SLA-enforced nodes: {len(analysis['sla_nodes'])}")
#         print(f"  - Decision gates: {analysis['decision_gates']}")
        
#     else:
#         print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")


async def execute_workflow(
    workflow_spec: WorkflowSpec,
    user_inputs: Optional[Dict[str, Any]] = None,
    objective: Optional[str] = None,
    conversation_id: Optional[str] = None,
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    debug: bool = False
) -> "WorkflowExecutionResult":
    """Execute a workflow spec and return results.
    
    This is the shared function used by both tests and web app for consistent execution.
    
    Args:
        workflow_spec: The workflow spec to execute
        user_inputs: Optional user inputs for data source nodes
        objective: Optional objective override, defaults to workflow description
        conversation_id: Optional conversation ID, will generate if not provided
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
    start_time = datetime.now()
    
    # Use workflow description as objective if not provided
    if objective is None:
        objective = workflow_spec.description or workflow_spec.title
    
    # Create initial state with user inputs
    initial_state = WorkflowState(
        initial_text=objective,
        conversation_id=conversation_id,
        user_inputs=user_inputs or {}
    )
    
    if debug:
        logger.info(f"🚀 Executing workflow: {workflow_spec.title}")
        logger.info(f"   Objective: {objective}")
        if user_inputs:
            logger.info(f"   User inputs: {list(user_inputs.keys())}")
    
    try:
        # Create DAG executor with typed execution
        executor = DAGExecutor(
            use_typed_execution=True,
            feedback_collector=feedback_collector
        )
        
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
        node_results = {}
        
        # Create a lookup for node types from the workflow spec
        node_type_lookup = {}
        for node in workflow_spec.nodes:
            node_type_lookup[node.id] = node.type
        
        for node_id, result in final_state.results.items():
            # Get actual node type from workflow spec
            actual_node_type = node_type_lookup.get(node_id, "agent")
            
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
                node_type=node_type,
                status=ExecutionStatus.COMPLETED,
                result=node_result
            )
        
        # Determine overall status - success means no failures, NOT all nodes executed!
        failed_count = stats.get("failed_nodes", 0)
        overall_status = ExecutionStatus.COMPLETED if failed_count == 0 else ExecutionStatus.PARTIAL
        
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=overall_status,
            node_results=node_results,
            final_output=final_state.results,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "stats": stats,
                "conversation_id": conversation_id
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


def create_default_tool_catalog(
    filter_broken: bool = True,
    verbose_format: bool = False, 
    use_working_filter: bool = True
) -> str:
    """Create the default tool catalog with standard settings.
    
    This ensures tests and web app use the same tool catalog configuration.
    
    Args:
        filter_broken: Filter out broken tools
        verbose_format: Use verbose format
        use_working_filter: Use working tools filter
        
    Returns:
        Tool catalog string
    """
    # Load tools first to ensure registry is populated
    load_tools_from_env()
    
    # Create catalog with same settings as web app
    return create_tool_catalog(
        filter_broken=filter_broken,
        verbose_format=verbose_format,
        use_working_filter=use_working_filter
    )


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


async def execute_workflow_with_metadata(
    workflow_spec: WorkflowSpec,
    execution_id: str,
    user_inputs: Optional[Dict[str, Any]] = None,
    form_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    client_mode: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
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
        Dict containing:
        - conversation_id: The conversation ID used
        - results: Final workflow state results dict
        - execution_stats: Execution statistics
        - success: Boolean indicating overall success
        - status: 'completed' or 'failed'
        - error: Optional error message if failed
    """
    conversation_id = conversation_id or f"execution_{execution_id}"
    
    if debug:
        logger.info(f"🛠️ Starting execution: {execution_id}")
        if user_inputs:
            logger.info(f"📝 User inputs provided: {list(user_inputs.keys())}")
        logger.info(f"📋 Executing workflow with {len(workflow_spec.nodes)} nodes")
    
    # Start feedback tracking if collector provided
    if feedback_collector:
        feedback_collector.start_execution_tracking(
            execution_id=execution_id,
            workflow_spec=workflow_spec,
            user_inputs=user_inputs or {}
        )
    
    try:
        # Create DAG executor with typed execution and feedback tracking
        executor = DAGExecutor(
            use_typed_execution=True,
            feedback_collector=feedback_collector
        )
        
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
        
        # Create initial state for DAG execution
        initial_state = WorkflowState(
            initial_text=workflow_spec.description,
            conversation_id=conversation_id,
            results={},
            user_inputs=user_inputs or {}
        )
        
        # Execute the DAG with the execution_id
        final_state = await executor.execute_dag(initial_state, execution_id=execution_id)
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        
        if debug:
            logger.info(f"✅ Execution completed: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
        
        return {
            "conversation_id": conversation_id,
            "results": final_state.results,
            "execution_stats": stats,
            "success": True,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {e}")
        
        # Complete feedback tracking with error if collector provided
        if feedback_collector:
            feedback_collector.complete_execution_tracking(
                execution_id=execution_id,
                final_outputs={},
                execution_summary={"error": str(e), "status": "failed"}
            )
        
        return {
            "conversation_id": conversation_id,
            "results": {},
            "execution_stats": None,
            "success": False,
            "status": "failed",
            "error": str(e)
        }