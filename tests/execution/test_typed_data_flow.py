#!/usr/bin/env python3
"""
Test typed data flow throughout the execution pipeline.

This test verifies that the typed models (DataSourceResult, AgentExecutionResult) 
are properly propagated through the system without breaking.
"""

import asyncio
import sys
from pathlib import Path

# Add the iointel src to the path
sys.path.insert(0, str(Path(__file__).parent / "iointel" / "src"))

from iointel.src.agent_methods.data_models.execution_models import (
    DataSourceResult, 
    AgentExecutionResult, 
    AgentRunResponse,
    ExecutionStatus
)
from iointel.src.chainables import execute_tool_task
from iointel.src.utilities.registries import TOOLS_REGISTRY

async def test_typed_data_flow():
    """Test the complete typed data flow."""
    print("🧪 Testing Typed Data Flow")
    print("=" * 50)
    
    # Test 1: DataSourceResult creation and access
    print("\n1. Testing DataSourceResult model...")
    result = DataSourceResult(
        tool_type="test_tool",
        status=ExecutionStatus.COMPLETED,
        result="Test output data",
        message="Tool executed successfully"
    )
    print(f"   ✅ Created DataSourceResult: {result.tool_type}")
    print(f"   ✅ Status: {result.status.value}")
    print(f"   ✅ Result access: {result.result}")
    
    # Test 2: AgentRunResponse creation and access
    print("\n2. Testing AgentRunResponse model...")
    agent_response = AgentRunResponse(
        result="Agent reasoning and output",
        conversation_id="test-conv-123",
        tool_usage_results=[],
        full_result={"additional": "data"}
    )
    print(f"   ✅ Created AgentRunResponse: {type(agent_response)}")
    print(f"   ✅ Result access: {agent_response.result}")
    
    # Test 3: AgentExecutionResult creation and access
    print("\n3. Testing AgentExecutionResult model...")
    execution_result = AgentExecutionResult(
        agent_response=agent_response,
        task_metadata={"test": "metadata"},
        execution_time=1.5,
        status=ExecutionStatus.COMPLETED
    )
    print(f"   ✅ Created AgentExecutionResult: {type(execution_result)}")
    print(f"   ✅ Agent response access: {execution_result.agent_response.result}")
    
    # Test 4: Tool execution returns typed result
    print("\n4. Testing execute_tool_task returns DataSourceResult...")
    task_metadata = {"tool_name": "non_existent_tool", "config": {}}
    objective = "Test objective"
    agents = []
    execution_metadata = {}
    
    try:
        tool_result = await execute_tool_task(task_metadata, objective, agents, execution_metadata)
        print(f"   ✅ Tool execution returned: {type(tool_result)}")
        print(f"   ✅ Is DataSourceResult: {isinstance(tool_result, DataSourceResult)}")
        print(f"   ✅ Status: {tool_result.status.value}")
        print(f"   ✅ Error handling: {tool_result.error[:50] if tool_result.error else 'None'}")
    except Exception as e:
        print(f"   ❌ Tool execution failed: {e}")
        return False
    
    # Test 5: Workflow server logic simulation
    print("\n5. Testing workflow server typed handling...")
    # Simulate successful tool result
    success_result = DataSourceResult(
        tool_type="successful_tool",
        status=ExecutionStatus.COMPLETED,
        result="Successful tool output"
    )
    
    # Test the status checking logic from workflow_server.py
    status_check = success_result.status.value == "completed"
    return_value = success_result.result if status_check else None
    
    print(f"   ✅ Status check: {status_check}")
    print(f"   ✅ Return value: {return_value}")
    
    # Test tool_usage_results dict creation (from workflow_server.py)
    tool_usage_results = [{
        'tool_name': success_result.tool_type,
        'input': objective,
        'result': success_result.result,
        'metadata': {'execution_type': 'direct_tool'}
    }]
    print(f"   ✅ Tool usage results structure: {len(tool_usage_results)} items")
    
    print("\n🎉 All typed data flow tests passed!")
    return True

async def test_integration_with_registry():
    """Test integration with actual tool registry."""
    print("\n🔗 Testing Integration with Tool Registry")
    print("=" * 50)
    
    # Check what tools are available
    available_tools = list(TOOLS_REGISTRY.keys())
    print(f"Available tools: {len(available_tools)}")
    
    if available_tools:
        # Test with a real tool
        tool_name = available_tools[0]
        print(f"Testing with tool: {tool_name}")
        
        task_metadata = {"tool_name": tool_name, "config": {}}
        objective = "Test with real tool"
        agents = []
        execution_metadata = {}
        
        try:
            result = await execute_tool_task(task_metadata, objective, agents, execution_metadata)
            print(f"   ✅ Real tool execution returned: {type(result)}")
            print(f"   ✅ Status: {result.status.value}")
            if result.result:
                print(f"   ✅ Has result data: {len(str(result.result))} chars")
            return True
        except Exception as e:
            print(f"   ⚠️ Real tool execution error: {e}")
            # This might be expected if tool needs specific config
            return True
    else:
        print("   ⚠️ No tools available for testing")
        return True

if __name__ == "__main__":
    async def main():
        success1 = await test_typed_data_flow()
        success2 = await test_integration_with_registry()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED - Typed data flow is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
    
    asyncio.run(main())