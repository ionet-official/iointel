"""
Quick test script for the WorkflowPlanner implementation.
"""

import asyncio
import json
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec


async def test_basic_functionality():
    """Test basic WorkflowPlanner functionality."""
    print("Testing WorkflowPlanner basic functionality...")
    
    # Create a simple example workflow first
    planner = WorkflowPlanner(debug=True)
    
    # Test example workflow creation
    example_workflow = planner.create_example_workflow("Test Workflow")
    print(f"✅ Created example workflow: {example_workflow.title}")
    print(f"   - Nodes: {len(example_workflow.nodes)}")
    print(f"   - Edges: {len(example_workflow.edges)}")
    
    # Validate the workflow spec structure
    assert isinstance(example_workflow, WorkflowSpec)
    assert example_workflow.title == "Test Workflow"
    assert len(example_workflow.nodes) > 0
    assert len(example_workflow.edges) > 0
    
    # Test JSON serialization
    workflow_json = example_workflow.model_dump_json(indent=2)
    print(f"✅ JSON serialization successful ({len(workflow_json)} characters)")
    
    # Test JSON deserialization
    workflow_from_json = WorkflowSpec.model_validate_json(workflow_json)
    assert workflow_from_json.id == example_workflow.id
    print("✅ JSON deserialization successful")
    
    return example_workflow


async def test_workflow_generation():
    """Test workflow generation from natural language (requires API key)."""
    print("\nTesting workflow generation...")
    
    # Skip if no API key available
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping workflow generation test (no OPENAI_API_KEY)")
        return
    
    planner = WorkflowPlanner(debug=True)
    
    # Simple tool catalog
    tool_catalog = {
        "fetch_data": {
            "description": "Fetch data from an API",
            "parameters": {"url": "string"},
            "returns": "data"
        },
        "send_email": {
            "description": "Send an email",
            "parameters": {"to": "string", "subject": "string", "body": "string"},
            "returns": "success"
        }
    }
    
    # Simple query
    query = "Fetch data from an API and send it via email"
    
    try:
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print(f"✅ Generated workflow: {workflow_spec.title}")
        print(f"   - Description: {workflow_spec.description}")
        print(f"   - Nodes: {len(workflow_spec.nodes)}")
        print(f"   - Node names: {workflow_spec.nodes}")
        print(f"   - Edges: {len(workflow_spec.edges)}")
        print(f"   - Edge names: {workflow_spec.edges}")
        
        return workflow_spec
        
    except Exception as e:
        print(f"❌ Workflow generation failed: {e}")
        return None


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test importing from the package
    try:
        from iointel.src.agent_methods.agents import WorkflowPlanner
        from iointel.src.agent_methods.data_models import WorkflowSpec, NodeSpec, EdgeSpec
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing WorkflowPlanner Implementation\n")
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed - stopping")
        return
    
    # Test basic functionality
    try:
        example_workflow = await test_basic_functionality()
        if example_workflow:
            print("✅ Basic functionality tests passed")
        else:
            print("❌ Basic functionality tests failed")
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
    
    # Test workflow generation
    try:
        generated_workflow = await test_workflow_generation()
        if generated_workflow:
            print("✅ Workflow generation tests passed")
        else:
            print("⚠️  Workflow generation tests skipped or failed")
    except Exception as e:
        print(f"❌ Workflow generation test error: {e}")
    
    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())