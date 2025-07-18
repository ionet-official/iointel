#!/usr/bin/env python3
"""
Test the Bitcoin Conditional Gate workflow to ensure it works with real tools.
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.web.workflow_storage import WorkflowStorage
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
from iointel.src.web.workflow_server import web_tool_executor, web_agent_executor

# Import tools to ensure they're registered
from iointel.src.agent_methods.tools.conditional_gate import conditional_gate
from iointel.src.agent_methods.tools.coinmarketcap import get_coin_quotes


async def test_bitcoin_workflow():
    """Test the Bitcoin Conditional Gate workflow."""
    
    # Load the workflow
    storage = WorkflowStorage()
    workflows = storage.list_workflows()
    
    # Find the Bitcoin workflow
    bitcoin_workflow = None
    for workflow in workflows:
        if "Bitcoin Conditional Gate Trading" in workflow["name"]:
            bitcoin_workflow_id = workflow["id"]
            spec = storage.load_workflow(bitcoin_workflow_id)
            bitcoin_workflow = spec
            break
    
    if not bitcoin_workflow:
        print("❌ Bitcoin Conditional Gate workflow not found!")
        return
    
    print(f"✅ Found workflow: {bitcoin_workflow.title}")
    print(f"   Description: {bitcoin_workflow.description}")
    print(f"   Nodes: {len(bitcoin_workflow.nodes)}")
    print(f"   Edges: {len(bitcoin_workflow.edges)}")
    
    # Register executors
    TASK_EXECUTOR_REGISTRY["tool"] = web_tool_executor
    TASK_EXECUTOR_REGISTRY["agent"] = web_agent_executor
    
    # Create DAG executor
    executor = DAGExecutor()
    executor.build_execution_graph(
        nodes=bitcoin_workflow.nodes,
        edges=bitcoin_workflow.edges,
        objective="Test Bitcoin conditional routing"
    )
    
    # Execute workflow
    print(f"\n🚀 Executing workflow...")
    initial_state = WorkflowState(
        conversation_id="test_bitcoin_workflow",
        initial_text="Test Bitcoin conditional routing workflow",
        results={}
    )
    
    try:
        final_state = await executor.execute_dag(initial_state)
        
        print(f"\n✅ Workflow execution completed!")
        print(f"   Results: {len(final_state.results)} nodes executed")
        
        # Show execution results
        for node_id, result in final_state.results.items():
            print(f"\n📋 Node: {node_id}")
            if isinstance(result, dict):
                if result.get("status") == "skipped":
                    print(f"   Status: SKIPPED (conditional routing worked!)")
                else:
                    print(f"   Status: EXECUTED")
                    print(f"   Result: {str(result)[:100]}...")
            else:
                print(f"   Status: EXECUTED")
                print(f"   Result: {str(result)[:100]}...")
        
        # Show execution statistics
        stats = executor.get_execution_statistics()
        print(f"\n📊 Execution Statistics:")
        print(f"   Total nodes: {stats['total_nodes']}")
        print(f"   Executed: {stats['executed_nodes']}")
        print(f"   Skipped: {stats['skipped_nodes']}")
        print(f"   Efficiency: {stats['execution_efficiency']}")
        
        if stats['skipped_nodes'] > 0:
            print(f"   🎯 SUCCESS: Conditional routing prevented unnecessary execution!")
            print(f"   Skipped nodes: {stats['skipped_node_ids']}")
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_bitcoin_workflow())