#!/usr/bin/env python3
"""
Simple test script for workflow helpers that can be run directly.
"""
import asyncio
import sys
import os
from uuid import uuid4

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

async def test_workflow_helpers():
    """Test the workflow helpers functionality."""
    print("🧪 Testing workflow helpers...")
    
    try:
        # Import the function we want to test
        from iointel.src.utilities.workflow_helpers import plan_and_execute
        
        # Test plan_and_execute with a simple math query
        result = await plan_and_execute(
            prompt="make a simple calculator agent with user input. YOU MUST USE CALCULATOR TOOLS TO SOLVE THE PROBLEM",
            conversation_id=str(uuid4()),
            debug=True
        )
        
        print("✅ Test completed!")
        print(f"   Status: {result.status}")
        print(f"   Workflow: {result.workflow_name}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   Nodes executed: {len(result.node_results)}")
        print(f"   Result: {result}")
        
        if result.final_output:
            print(f"   Final output keys: {list(result.final_output.keys())}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow_helpers()) 