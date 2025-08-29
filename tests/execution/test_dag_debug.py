#!/usr/bin/env python3
"""
Debug script to find where .ins is being accessed.
"""
import asyncio
import sys
import traceback
from uuid import uuid4

# Add better error tracking
import logging
logging.basicConfig(level=logging.DEBUG)

from iointel.src.utilities.workflow_helpers import execute_workflow
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    DataSourceNode,
    AgentNode,
    DataSourceData,
    AgentConfig,
    DataSourceConfig,
    EdgeSpec,
    SLARequirements
)

# Import tools to register them
from iointel.src.agent_methods.tools.basic_math import *

async def test_simple_workflow():
    """Test a simple workflow with the new structure."""
    workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        reasoning="Simple test workflow",
        title="Test Workflow",
        description="Test the new structure",
        nodes=[
            DataSourceNode(
                id="input1",
                type="data_source",
                label="Input",
                data=DataSourceData(
                    source_name="user_input",
                    config=DataSourceConfig(
                        message="Enter a value",
                        default_value="test"
                    )
                )
            ),
            AgentNode(
                id="agent1",
                type="agent",
                label="Agent",
                data=AgentConfig(
                    agent_instructions="Process the input",
                    tools=["add"]  # Using the actual registered tool name
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="e1",
                source="input1",
                target="agent1"
            )
        ]
    )
    
    print(f"Testing workflow: {workflow.title}")
    print(f"Nodes: {[n.id for n in workflow.nodes]}")
    
    # Check what attributes the AgentConfig has
    agent_node = workflow.nodes[1]
    print(f"\nAgentConfig attributes: {dir(agent_node.data)}")
    print(f"Has 'ins'? {hasattr(agent_node.data, 'ins')}")
    print(f"Has 'outs'? {hasattr(agent_node.data, 'outs')}")
    
    try:
        result = await execute_workflow(
            workflow_spec=workflow,
            user_inputs={"Input": "42"},
            debug=True
        )
        print(f"Result: {result.status}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to find the actual line causing the error
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback:
            print("\nTraceback frames:")
            for frame in traceback.extract_tb(exc_traceback):
                print(f"  File {frame.filename}, line {frame.lineno}, in {frame.name}")
                print(f"    {frame.line}")

if __name__ == "__main__":
    asyncio.run(test_simple_workflow())