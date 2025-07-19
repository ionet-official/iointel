#!/usr/bin/env python3
"""
Test script to simulate the user input flow and debug the issue.
"""

import asyncio
import json
import aiohttp
from pprint import pprint

async def test_user_input_flow():
    """Test the user input flow to debug the issue."""
    
    # Test workflow with user input
    workflow_data = {
        "title": "Test User Input Flow",
        "description": "Test workflow to debug user input data flow",
        "nodes": [
            {
                "id": "user_input_node",
                "type": "tool",
                "label": "Get User Input",
                "data": {
                    "tool_name": "user_input",
                    "config": {
                        "prompt": "What's your favorite number?",
                        "input_type": "text",
                        "placeholder": "Enter a number..."
                    },
                    "ins": [],
                    "outs": ["user_input"]
                }
            }
        ],
        "edges": []
    }
    
    # User input data - use the deterministic node ID
    user_inputs = {
        "user_input_1": "110000"
    }
    
    async with aiohttp.ClientSession() as session:
        # 1. Create workflow using generate endpoint
        print("1. Creating workflow...")
        generate_request = {
            "query": "Create a workflow that asks for user input using the user_input tool",
            "refine": False
        }
        async with session.post(
            "http://localhost:8000/api/generate",
            json=generate_request
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Workflow created: {result.get('success', False)}")
            else:
                print(f"❌ Failed to create workflow: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")
                return
        
        # 2. Execute workflow with user inputs
        print("\n2. Executing workflow with user inputs...")
        execution_request = {
            "execute_current": True,
            "user_inputs": user_inputs
        }
        
        async with session.post(
            "http://localhost:8000/api/execute",
            json=execution_request
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Execution started: {result.get('execution_id', 'No ID')}")
                execution_id = result.get('execution_id')
                
                # 3. Monitor execution
                if execution_id:
                    await asyncio.sleep(2)  # Wait for execution to complete
                    
                    async with session.get(
                        f"http://localhost:8000/api/executions/{execution_id}"
                    ) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            print(f"\n3. Execution status: {status_data.get('status', 'Unknown')}")
                            if status_data.get('results'):
                                print("Results:")
                                pprint(status_data['results'])
                            if status_data.get('error'):
                                print(f"Error: {status_data['error']}")
                        else:
                            print(f"❌ Failed to get execution status: {status_response.status}")
                
            else:
                print(f"❌ Failed to execute workflow: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")

if __name__ == "__main__":
    asyncio.run(test_user_input_flow())