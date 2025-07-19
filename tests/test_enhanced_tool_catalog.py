"""Test the enhanced tool catalog with workflow generation."""
import asyncio
import pytest
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.cli.run_workflow_planner import create_tool_catalog


@pytest.mark.asyncio
async def test_enhanced_tool_catalog_loading():
    """Test that the enhanced tool catalog loads correctly with rich parameter descriptions."""
    tool_catalog = create_tool_catalog()
    
    assert len(tool_catalog) > 0, "Tool catalog should not be empty"
    
    # Test a specific tool to see rich descriptions
    if 'get_coin_info' in tool_catalog:
        tool = tool_catalog['get_coin_info']
        assert 'name' in tool
        assert 'description' in tool
        assert 'parameters' in tool
        assert 'required_parameters' in tool
        
        # Check that parameters have rich descriptions
        for param_name, param_info in tool['parameters'].items():
            if isinstance(param_info, dict):
                assert 'type' in param_info
                assert 'description' in param_info
                assert 'required' in param_info


@pytest.mark.asyncio
async def test_enhanced_workflow_generation():
    """Test workflow generation with enhanced tool catalog."""
    tool_catalog = create_tool_catalog()
    planner = WorkflowPlanner(debug=False)
    
    # Test workflow generation
    workflow = await planner.generate_workflow(
        query="Create a workflow that gets Bitcoin price information and calculates the percentage change if it goes up by $5000",
        tool_catalog=tool_catalog
    )
    
    assert workflow is not None
    assert workflow.title
    assert workflow.description
    assert len(workflow.nodes) > 0
    
    # Check if workflow uses correct tools
    used_tools = []
    for node in workflow.nodes:
        if hasattr(node.data, 'tool_name') and node.data.tool_name:
            used_tools.append(node.data.tool_name)
    
    # Validate all tools exist in catalog
    invalid_tools = [tool for tool in used_tools if tool not in tool_catalog]
    assert len(invalid_tools) == 0, f"Invalid tools found: {invalid_tools}"


@pytest.mark.asyncio
async def test_complex_workflow_generation():
    """Test complex workflow generation that should use multiple tools/agents."""
    tool_catalog = create_tool_catalog()
    planner = WorkflowPlanner(debug=False)
    
    workflow = await planner.generate_workflow(
        query="Create a workflow that searches for recent news about Bitcoin, then gets Bitcoin price, and calculates potential profit from a $1000 investment",
        tool_catalog=tool_catalog
    )
    
    assert workflow is not None
    assert workflow.title
    assert workflow.description
    assert len(workflow.nodes) > 1, "Complex workflow should have multiple nodes"
    
    # Count node types
    tool_nodes = [n for n in workflow.nodes if n.type == 'tool']
    agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
    
    # Should have some combination of tools and agents
    assert len(tool_nodes) + len(agent_nodes) > 0
    
    # Validate tool node references (not agent tool lists, which may include hypothetical tools)
    used_tools = []
    for node in workflow.nodes:
        if node.type == 'tool' and hasattr(node.data, 'tool_name') and node.data.tool_name:
            used_tools.append(node.data.tool_name)
    
    invalid_tools = [tool for tool in used_tools if tool not in tool_catalog]
    assert len(invalid_tools) == 0, f"Invalid tool node tools found: {invalid_tools}"
    
    # Print workflow details for debugging
    print(f"\nGenerated workflow: {workflow.title}")
    print(f"Nodes: {len(workflow.nodes)}")
    for node in workflow.nodes:
        print(f"  - {node.id} ({node.type})")
        if hasattr(node.data, 'tool_name') and node.data.tool_name:
            print(f"    Tool: {node.data.tool_name}")
        if hasattr(node.data, 'tools') and node.data.tools:
            print(f"    Agent tools: {node.data.tools}")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_enhanced_tool_catalog_loading())
    asyncio.run(test_enhanced_workflow_generation())
    asyncio.run(test_complex_workflow_generation())
    print("✅ All enhanced tool catalog tests passed!")