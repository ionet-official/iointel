#!/usr/bin/env python3
"""
Debug script to test DAG executor with new WorkflowSpec structure.
"""
import asyncio
import traceback
from uuid import uuid4
from iointel.src.utilities.workflow_helpers import execute_workflow
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    DataSourceNode,
    AgentNode,
    DataSourceData,
    AgentConfig,
    DataSourceConfig,
    EdgeSpec
)

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
                    tools=["calculator_add"]
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
    
    try:
        result = await execute_workflow(
            workflow_spec=workflow,
            user_inputs={"Input": "42"},
            debug=True
        )
        print(f"Result: {result.status}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_workflow())