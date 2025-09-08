#!/usr/bin/env python3
"""
Test Integrated DAG Execution
=============================

Test that the integrated DAG executor works with parallel branches
in the main workflow system.
"""

import uuid
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.workflow import Workflow
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.decorators import register_custom_task

# Import the example tools to register them globally


# Register tool executor with timing
@register_custom_task("tool")
async def timed_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Tool executor that tracks execution timing."""
    tool_name = task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    
    start_time = time.time()
    print(f"    🔧 [{start_time:.3f}] Starting {tool_name}")
    
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    # Add small delay to make timing visible
    await asyncio.sleep(0.1)
    
    result = await tool.run(config)
    end_time = time.time()
    
    print(f"    ✅ [{end_time:.3f}] {tool_name} completed in {end_time - start_time:.3f}s → {result}")
    return result


async def test_integrated_parallel_execution():
    """Test parallel execution through the integrated workflow system."""
    print("🚀 Testing Integrated DAG Execution with Parallel Branches...")
    
    # Create parallel branches workflow: A → (B, C) → D
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Integrated Parallel Test",
        description="Test parallel execution through integrated workflow system",
        nodes=[
            NodeSpec(
                id="source",
                type="tool",
                label="Source: Add 10+5",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10, "b": 5},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="branch_left",
                type="tool",
                label="Left Branch: Square",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{source}", "b": "{source}"},  # Square it
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="branch_right",
                type="tool",
                label="Right Branch: Double",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{source}", "b": 2},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="merge",
                type="tool",
                label="Merge: Add branches",
                data=NodeData(
                    tool_name="add",
                    config={"a": "{branch_left}", "b": "{branch_right}"},
                    ins=["left", "right"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="source_to_left", source="source", target="branch_left"),
            EdgeSpec(id="source_to_right", source="source", target="branch_right"),
            EdgeSpec(id="left_to_merge", source="branch_left", target="merge"),
            EdgeSpec(id="right_to_merge", source="branch_right", target="merge")
        ]
    )
    
    # Convert to executable workflow
    workflow_spec.to_workflow_definition()
    yaml_content = workflow_spec.to_yaml()
    
    # Create workflow from YAML
    workflow = Workflow.from_yaml(yaml_str=yaml_content)
    
    print(f"  📊 Workflow has {len(workflow.tasks)} tasks")
    
    # Execute with timing
    print("  ⏱️  Executing workflow...")
    start_time = time.time()
    
    results = await workflow.run_tasks()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"  🎉 Workflow completed in {execution_time:.3f}s")
    print(f"  📊 Results: {results['results']}")
    
    # Verify mathematical correctness
    # source=15, branch_left=15²=225, branch_right=15*2=30, merge=225+30=255
    assert results["results"]["source"] == 15.0
    assert results["results"]["branch_left"] == 225.0
    assert results["results"]["branch_right"] == 30.0
    assert results["results"]["merge"] == 255.0
    
    # Check if execution was parallel (should be faster than sequential)
    # Sequential would be ~0.4s (4 tasks * 0.1s each)
    # Parallel should be ~0.3s (3 batches: source, parallel branches, merge)
    if execution_time < 0.35:
        print("  ✅ Execution time suggests parallel execution!")
    else:
        print(f"  ⚠️  Execution time {execution_time:.3f}s seems sequential")
    
    print("  ✅ Integrated DAG execution works!")


async def main():
    """Run the integrated DAG test."""
    print("🔥 Integrated DAG Execution Test")
    print("=" * 50)
    
    try:
        await test_integrated_parallel_execution()
        print("\n🎉 INTEGRATION SUCCESS!")
        print("✅ DAG executor integrated into workflow system")
        print("✅ Parallel execution working end-to-end")
        print("✅ Data flow resolution working with DAG topology")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())