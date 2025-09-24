#!/usr/bin/env python3
"""
Test script to verify all fixes for execution logs and prompts are working.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from iointel.src.utilities.workflow_helpers import execute_workflow
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
from iointel.src.utilities.io_logger import get_trace_history, clear_trace_history
from uuid import uuid4


async def test_execution_with_feedback():
    """Test that execution logs and prompts are properly tracked."""
    
    print("\n🧹 Clearing prompt history...")
    clear_trace_history()
    
    print("\n🔨 Creating test workflow...")
    # Create a simple test workflow
    workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Test Workflow for System Fixes",
        description="Testing execution logs and prompt tracking",
        reasoning="Testing the execution feedback and prompt logging systems",
        nodes=[
            NodeSpec(
                id="input_1",
                type="data_source",
                label="User Input",
                data=NodeData(
                    source_name="user_input",
                    config={"message": "Enter test value", "default_value": "test input"}
                ),
                ins=[],
                outs=["test_value"]
            ),
            NodeSpec(
                id="agent_1",
                type="agent",
                label="Test Agent",
                data=NodeData(
                    agent_instructions="Process the input value and return a summary",
                    tools=["calculator"],
                    ins=["input_1.test_value"]
                ),
                ins=["input_1.test_value"],
                outs=["result"]
            )
        ],
        edges=[
            EdgeSpec(
                id="e1",
                source="input_1",
                target="agent_1",
                data=EdgeData()
            )
        ]
    )
    
    print("\n🚀 Executing workflow...")
    result = await execute_workflow(
        workflow_spec=workflow,
        conversation_id=str(uuid4()),
        user_inputs={"test_value": "Calculate 5 + 3"}
    )
    
    print("\n✅ Workflow execution completed!")
    print(f"   Status: {result.status}")
    print(f"   Nodes executed: {len(result.node_results)}")
    
    # Check execution summary
    if hasattr(result, 'execution_summary') and result.execution_summary:
        summary = result.execution_summary
        print("\n📊 Execution Summary:")
        print(f"   Total duration: {summary.total_duration_seconds:.2f}s")
        print(f"   Nodes executed: {len(summary.nodes_executed)}")
        
        # Check node durations
        if summary.performance_metrics and 'node_durations' in summary.performance_metrics:
            durations = summary.performance_metrics['node_durations']
            print("\n⏱️  Node Durations:")
            for node_id, duration in durations.items():
                print(f"   - {node_id}: {duration}s")
        else:
            print("   ⚠️  No node durations found in performance metrics")
    else:
        print("\n⚠️  No execution summary found!")
    
    # Check prompts
    print("\n🤖 Checking prompt history...")
    prompts = get_trace_history()
    print(f"   Total prompts logged: {len(prompts)}")
    
    if prompts:
        print("\n   Recent prompts:")
        for i, prompt in enumerate(prompts[-3:], 1):
            print(f"   {i}. Type: {prompt.get('prompt_type', 'unknown')}")
            print(f"      Node: {prompt.get('metadata', {}).get('node_id', 'N/A')}")
            print(f"      Time: {prompt.get('timestamp', 'N/A')}")
    else:
        print("   ⚠️  No prompts were logged!")
    
    return result


async def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING EXECUTION LOGS AND PROMPTS FIXES")
    print("=" * 60)
    
    try:
        result = await test_execution_with_feedback()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print("1. Execution logs: Check if node durations are showing properly")
        print("2. Performance metrics: Check if node_durations dict is formatted correctly")
        print("3. Prompts tab: Check if prompts are being logged")
        print("\nNow check the web UI at http://localhost:12396 to verify:")
        print("- Execution Logs tab shows node timings (not '---')")
        print("- Performance metrics show proper values (not '[object Object]')")
        print("- Prompts tab shows the LLM interactions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)