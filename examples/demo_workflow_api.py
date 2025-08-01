#!/usr/bin/env python3
"""
Simple demonstration of Workflow API functionality
==================================================

This demonstrates the core concepts without multiprocessing complexity.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
from iointel.src.web.workflow_api_service import workflow_api_registry, WorkflowRunRequest

def create_simple_workflow() -> WorkflowSpec:
    """Create a simple workflow for demonstration."""
    
    # User input node
    user_input_node = NodeSpec(
        id="message_input",
        type="data_source", 
        label="Message Input",
        data=NodeData(
            config={
                "prompt": "Enter your message",
                "input_type": "text",
                "required": True
            },
            ins=[],
            outs=["user_message"],
            source_name="user_input",
            agent_instructions=None,
            tools=None,
            workflow_id=None,
            model="gpt-4o"
        )
    )
    
    # Processing agent
    processor_node = NodeSpec(
        id="message_processor",
        type="agent",
        label="Message Processor", 
        data=NodeData(
            config={},
            ins=["user_message"],
            outs=["processed_message"],
            tool_name=None,
            agent_instructions="You are a message processor. Take the user's message and respond with a friendly, helpful analysis of their message.",
            tools=None,
            workflow_id=None,
            model="gpt-4o"
        )
    )
    
    # Edge connecting them
    edge = EdgeSpec(
        id="e1",
        source="message_input",
        target="message_processor",
        sourceHandle=None,
        targetHandle=None,
        data=EdgeData(condition=None)
    )
    
    workflow_spec = WorkflowSpec(
        id=str(uuid.uuid4()),
        rev=1,
        reasoning="Simple workflow for API demonstration",
        title="Simple Message Processing Workflow",
        description="A workflow that processes user messages with AI analysis",
        nodes=[user_input_node, processor_node],
        edges=[edge],
        metadata={
            "tags": ["demo", "simple", "message-processing"],
            "complexity": "basic",
            "use_case": "demonstration"
        }
    )
    
    return workflow_spec

async def demonstrate_workflow_api():
    """Demonstrate the workflow API functionality."""
    
    print("🎯 WORKFLOW-AS-API DEMONSTRATION")
    print("=" * 50)
    
    # Demo organization structure
    org_id = "demo-org"
    user_id = "demo-user"
    workflow_id = "simple-message-processor"
    
    try:
        # 1. Create workflow specification
        print("\n📋 Step 1: Creating workflow specification...")
        workflow_spec = create_simple_workflow()
        print(f"✅ Created workflow: '{workflow_spec.title}'")
        print(f"   - Nodes: {len(workflow_spec.nodes)}")
        print(f"   - User input parameters: {len([n for n in workflow_spec.nodes if n.type == 'data_source' and getattr(n.data, 'source_name', None) == 'user_input'])}")
        
        # 2. Register workflow with API registry
        print(f"\n🔗 Step 2: Registering workflow...")
        registration_result = workflow_api_registry.register_workflow(
            org_id=org_id,
            user_id=user_id,
            workflow_id=workflow_id,
            workflow_spec=workflow_spec
        )
        print("✅ Workflow registered successfully!")
        print(f"   API endpoints created:")
        print(f"   - {registration_result['run_endpoint']}")
        print(f"   - {registration_result['spec_endpoint']}")
        
        # 3. Show user input parameter extraction
        registry_info = workflow_api_registry.registered_workflows[f"{org_id}/{user_id}/{workflow_id}"]
        user_params = registry_info["user_input_params"]
        print(f"\n📝 Step 3: Extracted user input parameters:")
        for param in user_params:
            print(f"   - {param['label']} (node_id: {param['node_id']})")
            print(f"     Type: {param['type']}, Required: {param['required']}")
        
        # 4. Demonstrate API execution with different input styles
        print(f"\n🚀 Step 4: Executing workflow via API...")
        
        # Simulate API call with inputs
        run_request = WorkflowRunRequest(
            inputs={"message": "Hello, can you analyze this message for sentiment?"},
            async_execution=False,
            metadata={"demo": True}
        )
        
        print(f"   Simulating API call:")
        print(f"   POST /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs")
        print(f"   Body: {json.dumps(run_request.model_dump(), indent=2)}")
        
        # Execute workflow (this would normally happen through HTTP)
        print(f"\n⚙️  Executing workflow...")
        execution_result = await workflow_api_registry.execute_workflow_api(
            org_id=org_id,
            user_id=user_id,
            workflow_id=workflow_id,
            run_request=run_request
        )
        
        print(f"✅ Execution completed!")
        print(f"   - Run ID: {execution_result.run_id}")
        print(f"   - Status: {execution_result.status}")
        print(f"   - Started: {execution_result.started_at}")
        if execution_result.completed_at:
            print(f"   - Completed: {execution_result.completed_at}")
        if execution_result.results:
            print(f"   - Results: {len(execution_result.results)} result objects")
        
        # 5. Show query parameter mapping
        print(f"\n🌐 Step 5: Query Parameter API Examples")
        print("   The following URL patterns would all work:")
        print(f"   GET /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/run?message=Hello")
        print(f"   GET /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/run?message_input=Hello")
        print(f"   GET /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/run?message=Hello&async_execution=false")
        
        # 6. Show cURL examples
        print(f"\n💻 Step 6: cURL Examples")
        print("   POST request:")
        print(f"   curl -X POST http://localhost:8001/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"inputs\": {\"message\": \"Hello world!\"}, \"async_execution\": false}'")
        print()
        print("   GET request:")
        print(f"   curl 'http://localhost:8001/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/run?message=Hello+world!'")
        
        # 7. Show production-ready features
        print(f"\n🏭 Step 7: Production Features")
        print("   ✅ Multi-tenant organization/user scoping")
        print("   ✅ Automatic user input → query parameter mapping")
        print("   ✅ Both synchronous and asynchronous execution")
        print("   ✅ Run status tracking and retrieval")
        print("   ✅ Workflow specification retrieval")
        print("   ✅ RESTful API design with proper HTTP methods")
        print("   ✅ JSON and query parameter input formats")
        
        print(f"\n🎉 Demonstration completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demonstrate_workflow_api())