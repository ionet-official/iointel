#!/usr/bin/env python3
"""
DAG Executor Testing
===================

Test the DAG execution engine to verify it properly handles:
1. True parallel execution of independent branches
2. Dependency ordering
3. Complex DAG topologies
4. Performance with timing verification
"""

import os
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
from iointel.src.utilities.dag_executor import DAGExecutor, create_dag_executor_from_spec
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.decorators import register_custom_task

# Import the example tools to register them globally
import iointel.src.RL.example_tools


# Register tool executor with timing
@register_custom_task("tool")
async def timed_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Tool executor that tracks execution timing."""
    tool_name = task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    
    start_time = time.time()
    print(f"    🔧 [{start_time:.3f}] Starting {tool_name} with config: {config}")
    
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    # Add small delay to simulate work and make timing visible
    await asyncio.sleep(0.1)
    
    result = await tool.run(config)
    end_time = time.time()
    
    print(f"    ✅ [{end_time:.3f}] {tool_name} completed in {end_time - start_time:.3f}s → {result}")
    return result


async def test_dag_executor_basic():
    """Test basic DAG executor functionality."""
    print("🧪 Testing Basic DAG Executor...")
    
    # Create simple linear chain: A → B → C
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Linear Chain DAG Test",
        description="Test basic DAG execution",
        nodes=[
            NodeSpec(
                id="step_a",
                type="tool",
                label="Step A",
                data=NodeData(
                    tool_name="add",
                    config={"a": 5, "b": 10},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="step_b",
                type="tool",
                label="Step B",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{step_a}", "b": 2},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="step_c",
                type="tool",
                label="Step C",
                data=NodeData(
                    tool_name="add",
                    config={"a": "{step_b}", "b": 1},
                    ins=["result"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="a_to_b", source="step_a", target="step_b"),
            EdgeSpec(id="b_to_c", source="step_b", target="step_c")
        ]
    )
    
    # Create executor and execute
    executor = create_dag_executor_from_spec(workflow_spec)
    
    # Verify execution plan
    summary = executor.get_execution_summary()
    print(f"  📊 Execution Summary: {summary}")
    
    assert summary["total_nodes"] == 3
    assert summary["total_batches"] == 3  # Each node in its own batch (linear)
    assert summary["max_parallelism"] == 1  # No parallelism in linear chain
    assert summary["execution_order"] == [["step_a"], ["step_b"], ["step_c"]]
    
    # Execute DAG
    initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    # Verify results
    assert final_state.results["step_a"] == 15.0  # 5 + 10
    assert final_state.results["step_b"] == 30.0  # 15 * 2
    assert final_state.results["step_c"] == 31.0  # 30 + 1
    
    print("  ✅ Basic DAG execution works!")


async def test_parallel_execution():
    """Test true parallel execution of independent branches."""
    print("\n🚀 Testing Parallel Execution...")
    
    # Create diamond pattern: A → (B, C) → D
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Parallel Execution Test",
        description="Test parallel branch execution",
        nodes=[
            NodeSpec(
                id="source",
                type="tool",
                label="Source",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10, "b": 5},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="branch_left",
                type="tool",
                label="Left Branch",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{source}", "b": 2},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="branch_right",
                type="tool",
                label="Right Branch", 
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{source}", "b": 3},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="merge",
                type="tool",
                label="Merge",
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
    
    # Create executor
    executor = create_dag_executor_from_spec(workflow_spec)
    
    # Verify execution plan shows parallelism
    summary = executor.get_execution_summary()
    print(f"  📊 Execution Summary: {summary}")
    
    assert summary["total_nodes"] == 4
    assert summary["total_batches"] == 3  # source → (left, right) → merge
    assert summary["max_parallelism"] == 2  # Two branches can run in parallel
    # The order within a batch may vary, so check structure not exact order
    assert len(summary["execution_order"]) == 3
    assert summary["execution_order"][0] == ["source"]
    assert set(summary["execution_order"][1]) == {"branch_left", "branch_right"}
    assert summary["execution_order"][2] == ["merge"]
    
    # Execute and time it
    print("  ⏱️  Timing parallel execution...")
    start_time = time.time()
    
    initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    assert final_state.results["source"] == 15.0      # 10 + 5
    assert final_state.results["branch_left"] == 30.0  # 15 * 2
    assert final_state.results["branch_right"] == 45.0 # 15 * 3
    assert final_state.results["merge"] == 75.0       # 30 + 45
    
    print(f"  ✅ Parallel execution completed in {execution_time:.3f}s")
    print(f"  📊 Results: {final_state.results}")
    
    # Verify branches ran in parallel (should be faster than sequential)
    # Each tool has 0.1s delay, so parallel should be ~0.3s, sequential would be ~0.4s
    assert execution_time < 0.35, f"Expected parallel execution < 0.35s, got {execution_time:.3f}s"
    
    print("  ✅ True parallel execution verified!")


async def test_complex_dag():
    """Test complex DAG with multiple levels and dependencies."""
    print("\n🌐 Testing Complex DAG...")
    
    # Create complex DAG: A → (B, C) → (D, E) → F
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Complex DAG Test",
        description="Test complex DAG topology",
        nodes=[
            NodeSpec(id="a", type="tool", label="A", data=NodeData(tool_name="add", config={"a": 1, "b": 2}, outs=["result"])),
            NodeSpec(id="b", type="tool", label="B", data=NodeData(tool_name="multiply", config={"a": "{a}", "b": 2}, outs=["result"])),
            NodeSpec(id="c", type="tool", label="C", data=NodeData(tool_name="multiply", config={"a": "{a}", "b": 3}, outs=["result"])),
            NodeSpec(id="d", type="tool", label="D", data=NodeData(tool_name="add", config={"a": "{b}", "b": 1}, outs=["result"])),
            NodeSpec(id="e", type="tool", label="E", data=NodeData(tool_name="add", config={"a": "{c}", "b": 2}, outs=["result"])),
            NodeSpec(id="f", type="tool", label="F", data=NodeData(tool_name="add", config={"a": "{d}", "b": "{e}"}, outs=["result"]))
        ],
        edges=[
            EdgeSpec(id="a_to_b", source="a", target="b"),
            EdgeSpec(id="a_to_c", source="a", target="c"),
            EdgeSpec(id="b_to_d", source="b", target="d"),
            EdgeSpec(id="c_to_e", source="c", target="e"),
            EdgeSpec(id="d_to_f", source="d", target="f"),
            EdgeSpec(id="e_to_f", source="e", target="f")
        ]
    )
    
    executor = create_dag_executor_from_spec(workflow_spec)
    summary = executor.get_execution_summary()
    
    print(f"  📊 Complex DAG Summary: {summary}")
    
    # Expected execution order: A → (B,C) → (D,E) → F
    # Order within batches may vary, so check structure
    assert len(summary["execution_order"]) == 4
    assert summary["execution_order"][0] == ["a"]
    assert set(summary["execution_order"][1]) == {"b", "c"}
    assert set(summary["execution_order"][2]) == {"d", "e"}
    assert summary["execution_order"][3] == ["f"]
    assert summary["max_parallelism"] == 2
    
    # Execute
    initial_state = WorkflowState(conversation_id="test", initial_text="", results={})
    final_state = await executor.execute_dag(initial_state)
    
    # Verify mathematical correctness
    # A=3, B=6, C=9, D=7, E=11, F=18
    expected = {"a": 3.0, "b": 6.0, "c": 9.0, "d": 7.0, "e": 11.0, "f": 18.0}
    for node_id, expected_val in expected.items():
        actual_val = final_state.results[node_id]
        assert actual_val == expected_val, f"Node {node_id}: expected {expected_val}, got {actual_val}"
    
    print("  ✅ Complex DAG executed correctly!")


async def test_dag_validation():
    """Test DAG validation catches errors."""
    print("\n🔍 Testing DAG Validation...")
    
    # Test cycle detection
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Cycle Test",
        description="Test cycle detection",
        nodes=[
            NodeSpec(id="a", type="tool", label="A", data=NodeData(tool_name="add", config={"a": 1, "b": 2})),
            NodeSpec(id="b", type="tool", label="B", data=NodeData(tool_name="add", config={"a": 1, "b": 2})),
            NodeSpec(id="c", type="tool", label="C", data=NodeData(tool_name="add", config={"a": 1, "b": 2}))
        ],
        edges=[
            EdgeSpec(id="a_to_b", source="a", target="b"),
            EdgeSpec(id="b_to_c", source="b", target="c"),
            EdgeSpec(id="c_to_a", source="c", target="a")  # Creates cycle!
        ]
    )
    
    executor = DAGExecutor()
    
    try:
        executor.build_execution_graph(workflow_spec.nodes, workflow_spec.edges)
        assert False, "Should have detected cycle"
    except ValueError as e:
        assert "Cycle detected" in str(e)
        print("  ✅ Cycle detection works!")
    
    print("  ✅ DAG validation works!")


async def main():
    """Run all DAG executor tests."""
    print("🚀 DAG Executor Comprehensive Testing")
    print("=" * 60)
    
    try:
        await test_dag_executor_basic()
        await test_parallel_execution()
        await test_complex_dag()
        await test_dag_validation()
        
        print("\n🎉 ALL DAG EXECUTOR TESTS PASSED!")
        print("✅ Basic DAG execution working")
        print("✅ True parallel execution verified")
        print("✅ Complex DAG topologies supported")
        print("✅ Cycle detection working")
        print("✅ Ready for integration into workflow system!")
        
    except Exception as e:
        print(f"\n❌ DAG EXECUTOR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())