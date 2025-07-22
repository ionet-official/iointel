#!/usr/bin/env python3
"""
Test Agent Data Flow Fix
========================

Test that the fix for agent data flow is working correctly.
This tests the specific scenario where user_input_node should pass
data to story_generation_agent.
"""

import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec, EdgeData
)
from iointel.src.workflow import Workflow


async def test_user_input_to_agent_workflow():
    """Test that user input flows correctly to agent task."""
    print("🧪 Testing User Input → Agent Data Flow...")
    
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="User Input to Agent Test",
        description="Test data flow from user input to agent task",
        nodes=[
            NodeSpec(
                id="user_input_node",
                type="agent",
                label="User Input Node",
                data=NodeData(
                    agent_instructions="Please provide the topic. Respond with just the topic, no additional text.",
                    outs=["topic"]
                )
            ),
            NodeSpec(
                id="story_generation_agent", 
                type="agent",
                label="Story Generation Agent",
                data=NodeData(
                    agent_instructions="Create a story based on the provided topic.",
                    ins=["topic"],
                    outs=["story"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="input_to_story", 
                source="user_input_node", 
                target="story_generation_agent",
                data=EdgeData(condition=None)
            )
        ]
    )
    
    # Convert to executable format
    workflow_spec.to_workflow_definition()
    yaml_content = workflow_spec.to_yaml()
    
    print(f"Workflow YAML:\n{yaml_content}")
    
    # Create workflow from YAML and execute
    workflow = Workflow.from_yaml(yaml_str=yaml_content)
    
    # Set the initial objective that should be passed to the first node
    workflow.objective = "alien abduction"
    
    # Execute the workflow
    result = await workflow.run_tasks()
    
    print("\n📊 Execution Results:")
    for task_id, output in result["results"].items():
        print(f"  {task_id}: {output}")
    
    # Verify both tasks completed
    assert "user_input_node" in result["results"], "user_input_node should have completed"
    assert "story_generation_agent" in result["results"], "story_generation_agent should have completed"
    
    # Verify the story agent got some meaningful output (not just asking for topic)
    story_output = result["results"]["story_generation_agent"]
    assert "Please provide the topic" not in story_output, "Agent should not be asking for topic anymore"
    assert len(story_output) > 50, "Story should be substantial"
    
    print("  ✅ Data flow from user input to agent works correctly!")
    return result


async def main():
    """Run the test."""
    print("🚀 Testing Agent Data Flow Fix")
    print("=" * 50)
    
    try:
        await test_user_input_to_agent_workflow()
        print("\n🎉 TEST PASSED!")
        print("Agent data flow issue has been fixed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(main())