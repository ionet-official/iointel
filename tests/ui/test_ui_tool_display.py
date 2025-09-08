#!/usr/bin/env python3
"""
Test script to demonstrate the UI tool usage display with mock data.
This shows how the enhanced UI will look when agents actually use tools.
"""

import json
from pathlib import Path


def create_mock_agent_result_with_tools():
    """Create mock agent result that includes tool usage results."""
    return {
        "math_solver": {
            "result": "I solved the problem step by step using calculator tools. First I added 12 + 8 = 20, then multiplied by 3 = 60, then found the square root ≈ 7.746.",
            "tool_usage_results": [
                {
                    "tool_name": "add",
                    "tool_args": {"a": 12, "b": 8},
                    "tool_result": 20.0
                },
                {
                    "tool_name": "multiply", 
                    "tool_args": {"a": 20.0, "b": 3},
                    "tool_result": 60.0
                },
                {
                    "tool_name": "square_root",
                    "tool_args": {"x": 60.0},
                    "tool_result": 7.745966692414834
                }
            ],
            "conversation_id": "mock_execution",
            "full_result": "Complete agent execution result"
        }
    }


def create_mock_workflow_spec():
    """Create a mock workflow spec with agent nodes that have tools."""
    return {
        "id": "mock-workflow-id",
        "title": "Math Solver with Tools Demo",
        "description": "Demonstrates agent using calculator tools",
        "nodes": [
            {
                "id": "math_solver",
                "type": "agent",
                "label": "Math Solver Agent",
                "data": {
                    "agent_instructions": "Solve mathematical problems using available calculator tools",
                    "tools": ["add", "multiply", "square_root"],
                    "config": {},
                    "ins": [],
                    "outs": ["result"]
                }
            }
        ],
        "edges": []
    }


def save_mock_execution_data():
    """Save mock execution data that can be loaded in the web UI."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Mock workflow results that show tool usage
    mock_results = {
        "results": create_mock_agent_result_with_tools(),
        "execution_id": "mock-execution-123",
        "status": "completed",
        "workflow_title": "Math Solver with Tools Demo"
    }
    
    # Mock workflow spec
    mock_workflow = create_mock_workflow_spec()
    
    # Save files
    with open(output_dir / "mock_execution_results.json", "w") as f:
        json.dump(mock_results, f, indent=2)
    
    with open(output_dir / "mock_workflow_spec.json", "w") as f:
        json.dump(mock_workflow, f, indent=2)
    
    print("📁 Mock data saved:")
    print("   - Execution results: output/mock_execution_results.json")
    print("   - Workflow spec: output/mock_workflow_spec.json")
    
    print("\n🎨 This demonstrates the UI enhancements:")
    print("   - Agent node shows tool pills: 🔧add 🔧multiply 🔧square_root")
    print("   - Execution results show structured tool usage:")
    print("     🛠️ Tool Usage Results: 3 tools used")
    print("       1. add - Args: {a: 12, b: 8} - Result: 20.0")
    print("       2. multiply - Args: {a: 20.0, b: 3} - Result: 60.0") 
    print("       3. square_root - Args: {x: 60.0} - Result: 7.746")
    
    return mock_results, mock_workflow


if __name__ == "__main__":
    print("🧪 UI Tool Display Demo")
    print("=" * 50)
    
    results, workflow = save_mock_execution_data()
    
    print("\n✅ Demo complete! The enhanced UI will show:")
    print("   - Rich panel-inspired tool usage display")
    print("   - Cyan-bordered tool panels with arguments and results")
    print("   - Tool pills showing available tools on agent nodes")
    print(f"   - Structured display of all {len(results['results']['math_solver']['tool_usage_results'])} tool executions")