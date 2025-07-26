"""
Workflow Helper Utilities
========================
High-level helpers for generating and executing workflows from natural language prompts.
"""
from typing import Dict, Any, Optional, List
from uuid import uuid4

from ..agent_methods.agents.workflow_planner import WorkflowPlanner
from ..agent_methods.data_models.workflow_spec import WorkflowSpec
from ..agent_methods.tools.tool_loader import load_tools_from_env
from ..utilities.tool_registry_utils import create_tool_catalog
from ..utilities.dag_executor import DAGExecutor
from ..utilities.graph_nodes import WorkflowState
from ..agent_methods.data_models.datamodels import AgentParams
from ..utilities.helpers import make_logger

logger = make_logger(__name__)


async def plan_and_execute(
    prompt: str,
    agents: Optional[List[AgentParams]] = None,
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
        agents: Optional list of AgentParams (auto-created if not provided)
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
    
    planner = WorkflowPlanner(
        conversation_id=conversation_id,
        debug=debug
    )
    
    if tool_catalog is None:
        # Load tools first to ensure registry is populated
        load_tools_from_env()
        tool_catalog = create_tool_catalog()
    
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
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=workflow_spec.nodes,
            edges=workflow_spec.edges,
            agents=agents,
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
    max_retries: int = 3,
    debug: bool = False
) -> Optional[WorkflowSpec]:
    """
    Just generate a workflow without executing it.
    Useful for testing workflow generation logic.
    
    Args:
        prompt: Natural language description
        tool_catalog: Optional tool catalog
        max_retries: Max generation retries
        debug: Enable debug logging
        
    Returns:
        WorkflowSpec or None if generation failed
    """
    planner = WorkflowPlanner(debug=debug)
    
    if tool_catalog is None:
        # Load tools first to ensure registry is populated
        load_tools_from_env()
        tool_catalog = create_tool_catalog()
    
    try:
        return await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries
        )
    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        return None


def analyze_workflow_spec(spec: WorkflowSpec) -> Dict[str, Any]:
    """
    Analyze a workflow spec for key features like execution modes, SLA, etc.
    
    Args:
        spec: WorkflowSpec to analyze
        
    Returns:
        Dict with analysis results
    """
    analysis = {
        "total_nodes": len(spec.nodes),
        "total_edges": len(spec.edges),
        "node_types": {},
        "execution_modes": {"consolidate": [], "for_each": []},
        "sla_nodes": [],
        "agents_with_tools": [],
        "conditional_edges": [],
        "decision_gates": []
    }
    
    for node in spec.nodes:
        # Count node types
        node_type = node.type
        analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1
        
        # Check execution mode
        if hasattr(node.data, 'execution_mode'):
            mode = node.data.execution_mode
            analysis["execution_modes"][mode].append(node.label)
        
        # Check for SLA
        if hasattr(node, 'sla') and node.sla:
            analysis["sla_nodes"].append({
                "node": node.label,
                "enforce_usage": getattr(node.sla, 'enforce_usage', False),
                "required_tools": getattr(node.sla, 'required_tools', []),
                "final_tool_must_be": getattr(node.sla, 'final_tool_must_be', None),
                "min_tool_calls": getattr(node.sla, 'min_tool_calls', 0)
            })
        
        # Check for agents with tools
        if node.type == "agent" and hasattr(node.data, 'tools') and node.data.tools:
            analysis["agents_with_tools"].append({
                "node": node.label,
                "tools": node.data.tools
            })
            
            # Check for conditional_gate
            if "conditional_gate" in node.data.tools:
                analysis["decision_gates"].append(node.label)
    
    # Check for conditional edges
    for edge in spec.edges:
        if edge.data and hasattr(edge.data, 'condition') and edge.data.condition:
            analysis["conditional_edges"].append({
                "source": edge.source,
                "target": edge.target,
                "condition": edge.data.condition
            })
    
    return analysis


# Convenience function for quick testing
async def quick_test(prompt: str) -> None:
    """
    Quick test function that generates, executes, and prints results.
    
    Args:
        prompt: Natural language workflow description
    """
    print(f"\n🧪 Testing: {prompt}")
    print("=" * 60)
    
    result = await plan_and_execute(prompt, debug=True)
    
    if result["success"]:
        spec = result["workflow_spec"]
        stats = result["execution_stats"]
        
        print("\n✅ SUCCESS!")
        print(f"📋 Workflow: {spec.title}")
        print(f"📊 Execution: {stats['executed_nodes']}/{stats['total_nodes']} nodes")
        
        # Analyze the workflow
        analysis = analyze_workflow_spec(spec)
        print("\n🔍 Analysis:")
        print(f"  - Node types: {analysis['node_types']}")
        print(f"  - For-each nodes: {analysis['execution_modes']['for_each']}")
        print(f"  - SLA-enforced nodes: {len(analysis['sla_nodes'])}")
        print(f"  - Decision gates: {analysis['decision_gates']}")
        
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")