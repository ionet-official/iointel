"""
Quick test script for the WorkflowPlanner implementation.
"""

import asyncio
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


async def test_agent_tool_integration():
    """Test agent-tool integration end-to-end."""
    print("\nTesting agent-tool integration...")
    
    # Load tools from creds.env and create catalog
    from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
    from iointel.src.agent_methods.workflow_converter import WorkflowConverter
    from iointel.src.utilities.tool_registry_utils import create_tool_catalog
    
    load_tools_from_env("creds.env")  # Load from creds.env
    tool_catalog = create_tool_catalog()
    print(f"📦 Loaded {len(tool_catalog)} tools")
    
    math_tools = [k for k in tool_catalog.keys() if "calculator" in k]
    print(f"🔢 Math tools available: {math_tools[:4]}...")  # Show first 4
    
    planner = WorkflowPlanner()
    
    try:
        # Generate workflow with agent that should use math tools
        workflow_spec = await planner.generate_workflow(
            query="Create a math agent that can solve arithmetic problems",
            tool_catalog=tool_catalog,
            max_retries=1
        )
        
        print(f"✅ Generated workflow: {workflow_spec.title}")
        print(f"📊 Nodes: {len(workflow_spec.nodes)}")
        
        # Check if agents have tools assigned
        agent_with_tools = False
        for node in workflow_spec.nodes:
            if node.type == "agent":
                print(f"🧠 Agent '{node.label}':")
                print(f"   Instructions: {node.data.agent_instructions}")
                if node.data.tools:
                    print(f"   Tools: {node.data.tools}")
                    agent_with_tools = True
                else:
                    print("   Tools: None")
                print(f"   Config: {node.data.config}")
        
        # Convert to executable workflow
        converter = WorkflowConverter()
        workflow_def = converter.convert(workflow_spec)
        print("🔄 Converted to WorkflowDefinition")
        
        # Check if agents got tools loaded
        for task in workflow_def.tasks:
            if task.agents and task.agents[0].tools:
                print(f"🔧 Task '{task.name}' agent loaded tools: {task.agents[0].tools}")
        
        if agent_with_tools:
            print("🎉 Agent-tool integration working!")
        else:
            print("⚠️ No agents with tools found")
        
        return workflow_spec
        
    except Exception as e:
        print(f"❌ Agent-tool integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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


async def test_agent_with_tools_execution():
    """Test agent-centric workflow with embedded tools and execution."""
    print("\nTesting agent with tools execution...")
    
    # Skip if no API key available
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping agent-tools execution test (no OPENAI_API_KEY)")
        return None, None

    # Load tools and create catalog
    from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
    from iointel.src.agent_methods.workflow_converter import WorkflowConverter
    from iointel.src.utilities.tool_registry_utils import create_tool_catalog
    from iointel.src.workflow import Workflow
    
    load_tools_from_env("creds.env")
    full_tool_catalog = create_tool_catalog()
    
    # Filter to only simple math tools from example_tools.py
    math_tool_names = ["add", "subtract", "multiply", "divide", "square_root"]
    tool_catalog = {name: tool for name, tool in full_tool_catalog.items() if name in math_tool_names}
    
    print(f"🔧 Loaded {len(full_tool_catalog)} total tools")
    print(f"🧮 Filtered to {len(tool_catalog)} math tools: {list(tool_catalog.keys())}")
    
    # Get model from environment
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"🤖 Using model: {model_name}")
    
    planner = WorkflowPlanner(debug=True)
    
    # Generate workflow with agent that MUST use math tools
    query = "Create a math solver agent that MUST use the calculator tools to solve step by step: 12 + 8, then multiply by 3, then find the square root. The agent must use actual tools, not mental math."
    
    try:
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog,
            max_retries=2
        )
        
        print(f"✅ Generated workflow: '{workflow_spec.title}'")
        print(f"📊 Nodes: {len(workflow_spec.nodes)}")
        
        # Check for agent nodes with tools
        agent_nodes = [node for node in workflow_spec.nodes if node.type == "agent"]
        print(f"🤖 Agent nodes: {len(agent_nodes)}")
        
        for node in agent_nodes:
            print(f"   - {node.label} (ID: {node.id})")
            print(f"     Instructions: {node.data.agent_instructions[:100]}...")
            print(f"     Tools: {node.data.tools}")
        
        # Convert to executable workflow
        converter = WorkflowConverter()
        workflow_def = converter.convert(workflow_spec)
        print("🔄 Converted to WorkflowDefinition")
        
        # Check if agents got tools loaded
        for task in workflow_def.tasks:
            if hasattr(task, 'agents') and task.agents and task.agents[0].tools:
                print(f"🔧 Task '{task.name}' agent tools: {task.agents[0].tools}")
        
        # Execute the workflow
        yaml_content = workflow_spec.to_yaml()
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        workflow.objective = workflow_spec.description
        
        print("🚀 Executing workflow...")
        conversation_id = "test_agent_tools_execution"
        results = await workflow.run_tasks(conversation_id=conversation_id)
        
        print("📊 Execution Results:")
        for task_id, result in results.get('results', {}).items():
            print(f"   Task: {task_id}")
            
            if isinstance(result, dict):
                # Check for tool usage results
                if 'tool_usage_results' in result:
                    print(f"     🛠️ Tool Usage Results: {len(result['tool_usage_results'])} tools used")
                    for i, tool_usage in enumerate(result['tool_usage_results']):
                        tool_name = tool_usage.get('tool_name', 'Unknown Tool')
                        tool_args = tool_usage.get('tool_args', {})
                        tool_result = tool_usage.get('tool_result', 'No result')
                        print(f"       {i+1}. {tool_name}")
                        print(f"          Args: {tool_args}")
                        print(f"          Result: {tool_result}")
                
                # Show main result
                main_result = result.get('result', result.get('output', str(result)))
                print(f"     Agent Output: {main_result}")
            else:
                print(f"     Result: {result}")
        
        # Check if we found tool usage results
        tool_usage_found = any(
            isinstance(result, dict) and 'tool_usage_results' in result 
            for result in results.get('results', {}).values()
        )
        
        if tool_usage_found:
            print("🎉 Agent-tool execution working! Tool usage results detected.")
        else:
            print("⚠️ No tool usage results found in agent output")
        
        return workflow_spec, results
        
    except Exception as e:
        print(f"❌ Agent-tools execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test importing from the package
    try:
        from iointel.src.agent_methods.agents import WorkflowPlanner
        from iointel.src.agent_methods.data_models import WorkflowSpec, NodeSpec, EdgeSpec
        print("✅ All imports successful")
        assert True  # Test passes if imports succeed
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Import failed: {e}"


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
    
    # Test agent-tool integration
    try:
        agent_tool_result = await test_agent_tool_integration()
        if agent_tool_result:
            print("✅ Agent-tool integration tests passed")
        else:
            print("❌ Agent-tool integration tests failed")
    except Exception as e:
        print(f"❌ Agent-tool integration test error: {e}")
    
    # Test workflow generation
    try:
        generated_workflow = await test_workflow_generation()
        if generated_workflow:
            print("✅ Workflow generation tests passed")
        else:
            print("⚠️  Workflow generation tests skipped or failed")
    except Exception as e:
        print(f"❌ Workflow generation test error: {e}")
    
    # Test agent with tools execution
    try:
        agent_workflow, agent_results = await test_agent_with_tools_execution()
        if agent_workflow and agent_results:
            print("✅ Agent-tool execution tests passed")
        else:
            print("⚠️  Agent-tool execution tests skipped or failed")
    except Exception as e:
        print(f"❌ Agent-tool execution test error: {e}")
    
    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())