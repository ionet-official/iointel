#!/usr/bin/env python3
"""
Simple test runner for Workflow API PoC
========================================

This creates a mock workflow to test the API without requiring the full Bitcoin workflow setup.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
import httpx
import uvicorn
from multiprocessing import Process
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData
from iointel.src.web.workflow_api_service import app as api_app

def create_mock_workflow() -> WorkflowSpec:
    """Create a simple mock workflow for testing."""
    
    # Create a simple workflow with user input and agent response
    user_input_node = NodeSpec(
        id="user_input",
        type="tool",
        label="User Input",
        data=NodeData(
            config={
                "prompt": "Enter your message",
                "input_type": "text",
                "required": True
            },
            ins=[],
            outs=["user_message"],
            tool_name="user_input",
            agent_instructions=None,
            tools=None,
            workflow_id=None,
            model="gpt-4o"
        )
    )
    
    echo_agent_node = NodeSpec(
        id="echo_agent",
        type="agent",
        label="Echo Agent",
        data=NodeData(
            config={},
            ins=["user_message"],
            outs=["echo_response"],
            tool_name=None,
            agent_instructions="You are an echo agent. Simply repeat back the user's message with a friendly greeting.",
            tools=None,
            workflow_id=None,
            model="gpt-4o"
        )
    )
    
    edge = EdgeSpec(
        id="e1",
        source="user_input",
        target="echo_agent",
        sourceHandle=None,
        targetHandle=None,
        data=None
    )
    
    workflow_spec = WorkflowSpec(
        id="test-echo-workflow",
        rev=1,
        reasoning="Simple test workflow for API demonstration",
        title="Echo Test Workflow",
        description="A simple workflow that echoes user input back",
        nodes=[user_input_node, echo_agent_node],
        edges=[edge],
        metadata={
            "tags": ["test", "demo", "echo"],
            "complexity": "simple",
            "use_case": "testing"
        }
    )
    
    return workflow_spec

async def test_workflow_api():
    """Test the workflow API with a simple mock workflow."""
    
    print("🧪 Testing Workflow API Service")
    print("=" * 40)
    
    # Start API server
    def run_server():
        uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="warning")
    
    server_process = Process(target=run_server)
    server_process.start()
    
    # Wait for server startup
    print("🚀 Starting API server...")
    time.sleep(2)
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            print("\n1. Testing health endpoint...")
            health_response = await client.get("http://localhost:8001/health")
            print(f"   Health status: {health_response.json()['status']}")
            
            # Create and register mock workflow
            print("\n2. Creating and registering mock workflow...")
            workflow_spec = create_mock_workflow()
            
            register_url = "http://localhost:8001/api/v1/orgs/test-org/users/test-user/workflows/echo-test/register"
            register_response = await client.post(
                register_url,
                json=workflow_spec.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            if register_response.status_code == 200:
                print("   ✅ Workflow registered successfully")
                registration_data = register_response.json()
                print(f"   Run endpoint: {registration_data['endpoints']['run_endpoint']}")
            else:
                print(f"   ❌ Registration failed: {register_response.status_code}")
                print(f"   Error: {register_response.text}")
            
            # Test GET execution with query parameters
            print("\n3. Testing GET execution with query parameters...")
            get_url = "http://localhost:8001/api/v1/orgs/test-org/users/test-user/workflows/echo-test/run"
            get_response = await client.get(
                get_url,
                params={
                    "message": "Hello from API test!",
                    "async_execution": "false"
                }
            )
            
            if get_response.status_code == 200:
                print("   ✅ GET execution successful")
                get_data = get_response.json()
                print(f"   Run ID: {get_data['run_id']}")
                print(f"   Status: {get_data['status']}")
            else:
                print(f"   ❌ GET execution failed: {get_response.status_code}")
                print(f"   Error: {get_response.text}")
            
            # Test POST execution with JSON body
            print("\n4. Testing POST execution with JSON body...")
            post_url = "http://localhost:8001/api/v1/orgs/test-org/users/test-user/workflows/echo-test/runs"
            post_response = await client.post(
                post_url,
                json={
                    "inputs": {"message": "Hello from POST request!"},
                    "async_execution": False
                }
            )
            
            if post_response.status_code == 200:
                print("   ✅ POST execution successful")
                post_data = post_response.json()
                print(f"   Run ID: {post_data['run_id']}")
                print(f"   Status: {post_data['status']}")
            else:
                print(f"   ❌ POST execution failed: {post_response.status_code}")
                print(f"   Error: {post_response.text}")
            
            # Test workflow spec retrieval
            print("\n5. Testing workflow spec retrieval...")
            spec_url = "http://localhost:8001/api/v1/orgs/test-org/users/test-user/workflows/echo-test/spec"
            spec_response = await client.get(spec_url)
            
            if spec_response.status_code == 200:
                print("   ✅ Spec retrieval successful")
                spec_data = spec_response.json()
                print(f"   Title: {spec_data['title']}")
                print(f"   Nodes: {len(spec_data['spec']['nodes'])}")
            else:
                print(f"   ❌ Spec retrieval failed: {spec_response.status_code}")
            
            print("\n🎉 API test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean shutdown
            print("\n🛑 Shutting down server...")
            server_process.terminate()
            server_process.join(timeout=3)
            if server_process.is_alive():
                server_process.kill()

if __name__ == "__main__":
    asyncio.run(test_workflow_api())