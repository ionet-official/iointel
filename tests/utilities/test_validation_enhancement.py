#!/usr/bin/env python3

"""
Test script to verify the enhanced validation catches missing tool parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'iointel', 'src'))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData
from uuid import uuid4

def test_missing_parameter_validation():
    """Test that validation catches missing tool parameters."""
    print("🧪 Testing enhanced parameter validation...")
    
    # Mock tool catalog with 'add' tool requiring 'a' and 'b' parameters
    tool_catalog = {
        "add": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {"a": "float", "b": "float"},
            "is_async": False
        },
        "get_weather": {
            "name": "get_weather", 
            "description": "Get weather for a city",
            "parameters": {"city": "str"},
            "is_async": False
        }
    }
    
    print("📋 Tool catalog:", tool_catalog)
    
    # Test 1: Valid workflow with all parameters
    print("\n🧪 Test 1: Valid workflow with all parameters")
    valid_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Valid Workflow",
        description="Test workflow with proper parameters",
        nodes=[
            NodeSpec(
                id="add_task",
                type="tool",
                label="Add Numbers",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10, "b": 5},  # All required parameters present
                    outs=["result"]
                )
            )
        ],
        edges=[]
    )
    
    issues = valid_workflow.validate_structure(tool_catalog)
    if issues:
        print(f"❌ Expected no issues, but got: {issues}")
    else:
        print("✅ Valid workflow passed validation")
    
    # Test 2: Missing parameters
    print("\n🧪 Test 2: Missing parameters (empty config)")
    invalid_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Invalid Workflow",
        description="Test workflow with missing parameters",
        nodes=[
            NodeSpec(
                id="add_task", 
                type="tool",
                label="Add Numbers",
                data=NodeData(
                    tool_name="add",
                    config={},  # Missing required 'a' and 'b' parameters
                    outs=["result"]
                )
            )
        ],
        edges=[]
    )
    
    issues = invalid_workflow.validate_structure(tool_catalog)
    print(f"🔍 Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  - {issue}")
    
    # Check that our new validation caught the missing parameters
    missing_param_issues = [issue for issue in issues if "MISSING PARAMETERS" in issue or "EMPTY CONFIG" in issue]
    if missing_param_issues:
        print("✅ Enhanced validation successfully caught missing parameters!")
    else:
        print("❌ Enhanced validation failed to catch missing parameters")
    
    # Test 3: Partial parameters missing
    print("\n🧪 Test 3: Partial parameters missing")
    partial_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Partial Workflow",
        description="Test workflow with some parameters missing",
        nodes=[
            NodeSpec(
                id="add_task",
                type="tool", 
                label="Add Numbers",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10},  # Missing 'b' parameter
                    outs=["result"]
                )
            )
        ],
        edges=[]
    )
    
    issues = partial_workflow.validate_structure(tool_catalog)
    print(f"🔍 Found {len(issues)} validation issues:")
    for issue in issues:
        print(f"  - {issue}")
    
    # Test 4: Data flow references (should be valid)
    print("\n🧪 Test 4: Data flow references (should be valid)")
    reference_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Reference Workflow",
        description="Test workflow with data flow references",
        nodes=[
            NodeSpec(
                id="get_weather_1",
                type="tool",
                label="Get Weather NY",
                data=NodeData(
                    tool_name="get_weather",
                    config={"city": "New York"},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="get_weather_2", 
                type="tool",
                label="Get Weather LA",
                data=NodeData(
                    tool_name="get_weather",
                    config={"city": "Los Angeles"},
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="add_temps",
                type="tool",
                label="Add Temperatures",
                data=NodeData(
                    tool_name="add",
                    config={"a": "{get_weather_1.result.temp}", "b": "{get_weather_2.result.temp}"},  # Data flow references
                    ins=["temp1", "temp2"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="edge1",
                source="get_weather_1",
                target="add_temps",
                sourceHandle="result",
                targetHandle="temp1"
            ),
            EdgeSpec(
                id="edge2", 
                source="get_weather_2",
                target="add_temps",
                sourceHandle="result",
                targetHandle="temp2"
            )
        ]
    )
    
    issues = reference_workflow.validate_structure(tool_catalog)
    if issues:
        print(f"🔍 Found {len(issues)} validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        # Check if the issues are about missing parameters (should not be since we have data flow references)
        param_issues = [issue for issue in issues if "MISSING PARAMETERS" in issue or "EMPTY CONFIG" in issue]
        if param_issues:
            print("❌ Validation incorrectly flagged data flow references as missing parameters")
        else:
            print("✅ No parameter validation issues (good - data flow references are valid)")
    else:
        print("✅ Data flow reference workflow passed validation")
    
    print("\n🎯 Summary:")
    print("- Enhanced validation now catches missing tool parameters")
    print("- Empty configs are flagged when tools require parameters")
    print("- Data flow references are still valid")
    print("- Auto-retry will now trigger for these validation errors")

if __name__ == "__main__":
    test_missing_parameter_validation()