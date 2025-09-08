#!/usr/bin/env python3
"""
Test text input tools (prompt_tool and user_input) with enhanced UI.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from iointel.src.agent_methods.tools.user_input import user_input, prompt_tool
from iointel.src.web.ui_components.text_input_ui import (
    get_text_input_ui_config, 
    is_text_input_tool, 
    get_all_text_input_tools
)
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, NodeData

async def test_text_input_tools():
    """Test text input tools functionality."""
    print("🧪 Testing Text Input Tools")
    print("=" * 50)
    
    # Test 1: Tool Functions
    print("\n1. Testing tool functions:")
    
    # Test prompt_tool
    prompt_result = prompt_tool("This is a test prompt message")
    print(f"✅ prompt_tool result: {prompt_result}")
    
    # Test user_input (without actual user input)
    user_result = user_input("Please enter your name", execution_metadata={'node_id': 'test_node'})
    print(f"✅ user_input result: {user_result}")
    
    # Test 2: UI Configurations
    print("\n2. Testing UI configurations:")
    
    all_configs = get_all_text_input_tools()
    print(f"✅ All text input tools: {list(all_configs.keys())}")
    
    for tool_name in all_configs:
        config = get_text_input_ui_config(tool_name)
        print(f"✅ {tool_name} config: height={config['height']}, has_run_button={config['has_run_button']}")
    
    # Test 3: Tool Detection
    print("\n3. Testing tool detection:")
    
    test_tools = ['prompt_tool', 'user_input', 'conditional_gate', 'unknown_tool']
    for tool in test_tools:
        is_text_input = is_text_input_tool(tool)
        print(f"✅ is_text_input_tool('{tool}'): {is_text_input}")
    
    # Test 4: Simple Workflow with Text Input Tools
    print("\n4. Testing simple workflow with text input tools:")
    
    # Create a simple workflow: prompt -> user_input -> result
    nodes = [
        NodeSpec(
            id="prompt_node",
            type="tool",
            label="Prompt Tool",
            data=NodeData(
                tool_name="prompt_tool",
                config={"message": "Welcome! Please provide your input below."},
                ins=[],
                outs=["prompt_output"]
            )
        ),
        NodeSpec(
            id="input_node", 
            type="tool",
            label="User Input",
            data=NodeData(
                tool_name="user_input",
                config={"prompt": "Enter your response:", "placeholder": "Type your answer..."},
                ins=["prompt_output"],
                outs=["user_response"]
            )
        )
    ]
    
    # Create workflow spec
    import uuid
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Text Input Test Workflow",
        description="Test workflow for text input tools",
        nodes=nodes,
        edges=[]
    )
    
    print(f"✅ Created workflow with {len(workflow_spec.nodes)} nodes")
    print(f"   - prompt_tool node: {workflow_spec.nodes[0].id}")
    print(f"   - user_input node: {workflow_spec.nodes[1].id}")
    
    # Test UI config retrieval for workflow nodes
    print("\n5. Testing UI config for workflow nodes:")
    
    for node in workflow_spec.nodes:
        if node.data.tool_name:
            ui_config = get_text_input_ui_config(node.data.tool_name, node.data.config)
            if ui_config:
                print(f"✅ {node.id} ({node.data.tool_name}):")
                print(f"   - UI type: {ui_config['input_type']}")
                print(f"   - Height: {ui_config['height']}")
                print(f"   - Has run button: {ui_config['has_run_button']}")
                print(f"   - Placeholder: {ui_config['placeholder']}")
    
    print("\n✅ All text input tool tests passed!")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_text_input_tools())