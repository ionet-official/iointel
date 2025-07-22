#!/usr/bin/env python3
"""
Comprehensive Data Flow Resolution Tests
=======================================

Test the data flow resolution system with increasingly complex scenarios
to ensure we haven't missed any edge cases or architectural gaps.
"""

import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.utilities.data_flow_resolver import data_flow_resolver
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.decorators import register_custom_task
from iointel.src.workflow import Workflow

# Import the example tools to register them globally


# Register tool executor for testing
@register_custom_task("tool")
async def test_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Test tool executor with detailed logging."""
    tool_name = task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    
    print(f"    🔧 Executing {tool_name} with config: {config}")
    
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    result = await tool.run(config)
    print(f"    ✅ {tool_name} → {result}")
    return result


def test_data_flow_resolver_unit_tests():
    """Test the DataFlowResolver directly with various scenarios."""
    print("🧪 Unit Testing DataFlowResolver...")
    
    # Test 1: Simple variable resolution
    print("\n  Test 1: Simple variable resolution")
    results = {"node_a": 42, "node_b": {"value": 100, "status": "ok"}}
    config = {"x": "{node_a}", "y": "{node_b.value}", "z": 10}
    resolved = data_flow_resolver.resolve_config(config, results)
    assert resolved == {"x": 42, "y": 100, "z": 10}
    print("    ✅ Simple resolution works")
    
    # Test 2: Multiple references in one string
    print("\n  Test 2: Template strings with multiple references")
    config = {"message": "Result: {node_a} + {node_b.value} = {total}"}
    results = {"node_a": 42, "node_b": {"value": 100}, "total": 142}
    resolved = data_flow_resolver.resolve_config(config, results)
    assert resolved["message"] == "Result: 42 + 100 = 142"
    print("    ✅ Template strings work")
    
    # Test 3: Nested field access
    print("\n  Test 3: Nested field access")
    results = {"fetch_api": {"response": {"data": {"temperature": 72.5}}}}
    config = {"temp": "{fetch_api.response.data.temperature}"}
    resolved = data_flow_resolver.resolve_config(config, results)
    assert resolved["temp"] == 72.5
    print("    ✅ Nested field access works")
    
    # Test 4: Error handling - missing node
    print("\n  Test 4: Error handling - missing node")
    try:
        config = {"x": "{missing_node}"}
        data_flow_resolver.resolve_config(config, results)
        assert False, "Should have thrown error"
    except ValueError as e:
        assert "missing_node" in str(e)
        print("    ✅ Missing node error handled")
    
    # Test 5: Error handling - missing field
    print("\n  Test 5: Error handling - missing field")
    try:
        config = {"x": "{node_a.missing_field}"}
        results = {"node_a": 42}
        data_flow_resolver.resolve_config(config, results)
        assert False, "Should have thrown error"
    except ValueError as e:
        assert "missing_field" in str(e)
        print("    ✅ Missing field error handled")
    
    print("  ✅ All unit tests passed!")


async def test_linear_chain_workflow():
    """Test a simple linear chain: A → B → C → D."""
    print("\n🔗 Testing Linear Chain Workflow...")
    
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Linear Chain Test",
        description="Test A→B→C→D data flow",
        nodes=[
            NodeSpec(
                id="step_a",
                type="tool",
                label="Step A: Add 10+5",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10, "b": 5},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="step_b", 
                type="tool",
                label="Step B: Multiply by 2",
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
                label="Step C: Square root",
                data=NodeData(
                    tool_name="square_root",
                    config={"x": "{step_b}"},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="step_d",
                type="tool",
                label="Step D: Add 1",
                data=NodeData(
                    tool_name="add",
                    config={"a": "{step_c}", "b": 1},
                    ins=["result"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="a_to_b", source="step_a", target="step_b"),
            EdgeSpec(id="b_to_c", source="step_b", target="step_c"),
            EdgeSpec(id="c_to_d", source="step_c", target="step_d"),
        ]
    )
    
    # Expected: (10+5)*2 = 30, √30 ≈ 5.477, 5.477+1 ≈ 6.477
    result = await execute_workflow_spec(workflow_spec)
    
    assert "step_a" in result["results"]
    assert "step_b" in result["results"] 
    assert "step_c" in result["results"]
    assert "step_d" in result["results"]
    
    # Verify mathematical correctness
    assert result["results"]["step_a"] == 15.0
    assert result["results"]["step_b"] == 30.0
    assert abs(result["results"]["step_c"] - 5.477225575051661) < 0.001
    assert abs(result["results"]["step_d"] - 6.477225575051661) < 0.001
    
    print("  ✅ Linear chain executed correctly!")
    print(f"    step_a: {result['results']['step_a']}")
    print(f"    step_b: {result['results']['step_b']}")
    print(f"    step_c: {result['results']['step_c']}")
    print(f"    step_d: {result['results']['step_d']}")


async def test_parallel_branches_workflow():
    """Test parallel branches that later merge: A → B,C → D."""
    print("\n🌳 Testing Parallel Branches Workflow...")
    
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Parallel Branches Test",
        description="Test A→(B,C)→D parallel execution",
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
                id="branch_1",
                type="tool",
                label="Branch 1: Square",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{source}", "b": "{source}"},  # Square it
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="branch_2", 
                type="tool",
                label="Branch 2: Double",
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
                    config={"a": "{branch_1}", "b": "{branch_2}"},
                    ins=["result1", "result2"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="source_to_b1", source="source", target="branch_1"),
            EdgeSpec(id="source_to_b2", source="source", target="branch_2"),
            EdgeSpec(id="b1_to_merge", source="branch_1", target="merge"),
            EdgeSpec(id="b2_to_merge", source="branch_2", target="merge"),
        ]
    )
    
    # Expected: source=15, branch_1=15²=225, branch_2=15*2=30, merge=225+30=255
    result = await execute_workflow_spec(workflow_spec)
    
    assert result["results"]["source"] == 15.0
    assert result["results"]["branch_1"] == 225.0  # 15²
    assert result["results"]["branch_2"] == 30.0   # 15*2  
    assert result["results"]["merge"] == 255.0     # 225+30
    
    print("  ✅ Parallel branches executed correctly!")
    print(f"    source: {result['results']['source']}")
    print(f"    branch_1: {result['results']['branch_1']}")
    print(f"    branch_2: {result['results']['branch_2']}")
    print(f"    merge: {result['results']['merge']}")


async def test_complex_field_access():
    """Test complex field access patterns."""
    print("\n🎯 Testing Complex Field Access...")
    
    # Create a workflow that uses weather tool which returns structured data
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Complex Field Access Test",
        description="Test accessing nested fields from tool results",
        nodes=[
            NodeSpec(
                id="get_weather",
                type="tool",
                label="Get Weather Data",
                data=NodeData(
                    tool_name="get_weather",
                    config={"city": "New York"},
                    outs=["weather_data"]
                )
            ),
            NodeSpec(
                id="check_temp",
                type="tool",
                label="Add 10 to temperature",
                data=NodeData(
                    tool_name="add",
                    config={"a": "{get_weather.temp}", "b": 10},  # Access .temp field
                    ins=["weather_data"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="weather_to_temp", source="get_weather", target="check_temp")
        ]
    )
    
    result = await execute_workflow_spec(workflow_spec)
    
    # Weather tool returns {"temp": ~72.x, "condition": "Sunny"}
    weather_data = result["results"]["get_weather"]
    temp_plus_10 = result["results"]["check_temp"]
    
    print(f"    weather_data: {weather_data}")
    print(f"    temp_plus_10: {temp_plus_10}")
    
    # Verify field access worked
    assert "temp" in weather_data
    assert abs(temp_plus_10 - (weather_data["temp"] + 10)) < 0.001
    
    print("  ✅ Complex field access works!")


async def execute_workflow_spec(workflow_spec: WorkflowSpec) -> dict:
    """Execute a WorkflowSpec and return results."""
    # Convert to executable format
    workflow_spec.to_workflow_definition()
    yaml_content = workflow_spec.to_yaml()
    
    # Create workflow from YAML and execute
    workflow = Workflow.from_yaml(yaml_str=yaml_content)
    return await workflow.run_tasks()


async def main():
    """Run all comprehensive tests."""
    print("🚀 Comprehensive Data Flow Resolution Testing")
    print("=" * 60)
    
    try:
        # Unit tests first
        test_data_flow_resolver_unit_tests()
        
        # Integration tests
        await test_linear_chain_workflow()
        await test_parallel_branches_workflow() 
        await test_complex_field_access()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Data flow resolution is working correctly across all scenarios!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())