#!/usr/bin/env python3
"""
Proof of Concept: Workflows as APIs
===================================

This script demonstrates how to turn the Bitcoin trading workflow into a bespoke API endpoint.
It shows the complete WaaS (Workflow as a Service) workflow:

1. Load Bitcoin workflow specification
2. Register it with the API service
3. Make API calls with different query parameters
4. Show how user inputs become URL parameters

Usage:
    uv run workflow_api_poc.py
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
import signal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.web.workflow_api_service import app as api_app

def run_server():
    uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="info")

class WorkflowAPIDemo:
    """Demo class for workflow-as-API functionality."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8001"
        self.client = httpx.AsyncClient()
        self.server_process = None
        
        # Demo organization/user structure
        self.org_id = "io-intel"
        self.user_id = "test-user"
        self.workflow_id = "bitcoin-trading"
    
    def start_api_server(self):
        """Start the API server in a separate process."""
        self.server_process = Process(target=run_server)
        self.server_process.start()
        
        # Wait for server to start
        print("🚀 Starting API server...")
        time.sleep(3)
        
        # Verify server is running
        try:
            response = httpx.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                print("✅ API server is running")
            else:
                print(f"❌ Server health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
    
    def stop_api_server(self):
        """Stop the API server."""
        if self.server_process is not None:
            print("🛑 Stopping API server...")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.kill()
        else:
            print("[stop_api_server] No server process to terminate.")
    
    def load_bitcoin_workflow(self) -> WorkflowSpec:
        """Load the Bitcoin trading workflow specification."""
        workflow_path = project_root / "saved_workflows" / "json" / "Bitcoin Conditional Gate Trading_2a308f54.json"
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Bitcoin workflow not found at {workflow_path}")
        
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        # Convert to WorkflowSpec
        workflow_spec = WorkflowSpec(**workflow_data)
        return workflow_spec
    
    async def register_workflow(self, workflow_spec: WorkflowSpec) -> Dict[str, Any]:
        """Register the workflow with the API service."""
        url = f"{self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/register"
        
        response = await self.client.post(
            url,
            json=workflow_spec.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to register workflow: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def execute_workflow_post(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow using POST request with JSON body."""
        url = f"{self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/runs"
        
        request_body = {
            "inputs": inputs or {},
            "async_execution": False,  # Synchronous for demo
            "metadata": {"demo": True}
        }
        
        response = await self.client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to execute workflow: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def execute_workflow_get(self, **query_params) -> Dict[str, Any]:
        """Execute workflow using GET request with query parameters."""
        url = f"{self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/run"
        
        response = await self.client.get(url, params=query_params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to execute workflow: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def get_workflow_spec(self) -> Dict[str, Any]:
        """Get the workflow specification via API."""
        url = f"{self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/spec"
        
        response = await self.client.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get workflow spec: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def run_demo(self):
        """Run the complete workflow-as-API demonstration."""
        try:
            print("=" * 60)
            print("🚀 WORKFLOW-AS-API PROOF OF CONCEPT")
            print("=" * 60)
            
            # 1. Load workflow specification
            print("\n📋 Step 1: Loading Bitcoin trading workflow...")
            workflow_spec = self.load_bitcoin_workflow()
            print(f"✅ Loaded workflow: '{workflow_spec.title}'")
            print(f"   - Nodes: {len(workflow_spec.nodes)}")
            print(f"   - Edges: {len(workflow_spec.edges)}")
            
            # 2. Register workflow with API service
            print("\n🔗 Step 2: Registering workflow with API service...")
            registration_result = await self.register_workflow(workflow_spec)
            print("✅ Workflow registered successfully!")
            print(f"   - Run endpoint: {registration_result['endpoints']['run_endpoint']}")
            print(f"   - Spec endpoint: {registration_result['endpoints']['spec_endpoint']}")
            
            # 3. Get workflow spec via API
            print("\n📖 Step 3: Retrieving workflow spec via API...")
            spec_response = await self.get_workflow_spec()
            print(f"✅ Retrieved spec: {spec_response['title']}")
            print(f"   - Description: {spec_response['description']}")
            
            # 4. Execute workflow using POST with JSON body
            print("\n🔄 Step 4: Executing workflow via POST (JSON body)...")
            post_result = await self.execute_workflow_post({
                "symbol": "BTC",
                "market": "USD"
            })
            print("✅ Execution started!")
            print(f"   - Run ID: {post_result['run_id']}")
            print(f"   - Status: {post_result['status']}")
            if post_result.get('results'):
                print(f"   - Results: {json.dumps(post_result['results'], indent=2)[:200]}...")
            
            # 5. Execute workflow using GET with query parameters
            print("\n🌐 Step 5: Executing workflow via GET (query parameters)...")
            print("   URL: GET /api/v1/orgs/io-intel/users/test-user/workflows/bitcoin-trading/run?symbol=ETH&market=USD")
            
            get_result = await self.execute_workflow_get(
                symbol="ETH",
                market="USD",
                async_execution="false"
            )
            print("✅ Execution completed!")
            print(f"   - Run ID: {get_result['run_id']}")
            print(f"   - Status: {get_result['status']}")
            if get_result.get('results'):
                print(f"   - Results available: {len(get_result['results'])} result objects")
            
            # 6. Show different API usage patterns
            print("\n💡 Step 6: API Usage Examples")
            print("=" * 40)
            print("POST Request (JSON body):")
            print(f"POST {self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/runs")
            print('{"inputs": {"symbol": "BTC"}, "async_execution": false}')
            print()
            print("GET Request (Query parameters):")
            print(f"GET {self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/run?symbol=BTC&async_execution=false")
            print()
            print("cURL Examples:")
            print(f"curl -X POST {self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/runs \\")
            print('     -H "Content-Type: application/json" \\')
            print('     -d \'{"inputs": {"symbol": "BTC"}, "async_execution": false}\'')
            print()
            print(f"curl {self.api_base_url}/api/v1/orgs/{self.org_id}/users/{self.user_id}/workflows/{self.workflow_id}/run?symbol=BTC")
            
            print("\n🎉 Demo completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.client.aclose()

async def main():
    """Main demo function."""
    demo = WorkflowAPIDemo()
    
    # Setup signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal, shutting down...")
        demo.stop_api_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start API server
        demo.start_api_server()
        
        # Run demo
        await demo.run_demo()
        
    finally:
        # Clean shutdown
        demo.stop_api_server()

if __name__ == "__main__":
    asyncio.run(main())