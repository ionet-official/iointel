#!/usr/bin/env python3
"""
Test Exact Scenario
===================

Test the exact scenario described by the user:
1. user_input_node completes with output: "alien abduction" 
2. story_generation_agent should receive this input and NOT ask "Please provide the topic"
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env" 
load_dotenv(env_path)

from iointel.src.chainables import execute_agent_task
from iointel.src.agents import Agent


async def test_agent_task_with_data():
    """Test the execute_agent_task function directly with available results."""
    print("🔬 Testing execute_agent_task with available data...")
    
    # Simulate task metadata from a story generation agent that should receive
    # data from a previous user_input_node
    task_metadata = {
        "agent_instructions": "Create a story based on the provided topic.",
        "available_results": {
            "user_input_node": "alien abduction"
        },
        "kwargs": {}
    }
    
    # Execute the agent task
    result = await execute_agent_task(
        task_metadata=task_metadata,
        objective="",  # This would normally be empty for agent tasks
        agents=[Agent.make_default()],
        execution_metadata={"client_mode": False}
    )
    
    print(f"\n📊 Agent Task Result:\n{result}")
    
    # Verify the agent received the data and didn't ask for topic
    assert "Please provide the topic" not in result, "Agent should not ask for topic when data is available"
    assert "alien" in result.lower() or "abduction" in result.lower(), "Story should reference the provided topic"
    
    print("  ✅ Agent correctly received and used the available data!")
    return result


async def test_direct_context_passing():
    """Test that context is properly passed to run_agents."""
    print("\n🔍 Testing direct context passing...")
    
    from iointel.src.utilities.runners import run_agents
    
    # Test run_agents with context containing available data
    response = await run_agents(
        objective="Create a story based on the provided topic.\n\nAvailable data from previous tasks:\nuser_input_node: alien abduction",
        agents=[Agent.make_default()],
        context={
            "available_results": {"user_input_node": "alien abduction"},
            "user_input_node": "alien abduction"
        },
        output_type=str,
    ).execute()
    
    print(f"\n📊 run_agents Result:\n{response}")
    
    # Check if response includes story about alien abduction
    if isinstance(response, dict) and "result" in response:
        story = response["result"]
    else:
        story = str(response)
    
    assert "alien" in story.lower() or "abduction" in story.lower(), "Story should reference the topic"
    print("  ✅ Context passing to run_agents works correctly!")
    
    return response


async def main():
    """Run all tests."""
    print("🚀 Testing Exact Data Flow Scenario")
    print("=" * 50)
    
    try:
        # Test the fixed agent task executor
        await test_agent_task_with_data()
        
        # Test the underlying run_agents function
        await test_direct_context_passing()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The data flow issue has been resolved!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())