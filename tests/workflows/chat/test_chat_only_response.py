"""
Test to verify chat-only response handling in workflow server.
"""

import asyncio
from unittest.mock import Mock, patch
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpecLLM, WorkflowSpec
from iointel.src.web.workflow_server import generate_workflow
from iointel.src.web.workflow_server import WorkflowRequest


async def test_chat_only_response():
    """Test that chat-only responses don't cause 'reading rev' errors."""
    
    # Create a mock chat-only response
    chat_only_response = WorkflowSpecLLM(
        title=None,
        description="I need more information about what kind of workflow you want.",
        reasoning="The user asked about available tools, which requires a conversational response.",
        nodes=None,
        edges=None
    )
    
    # Mock the workflow planner to return chat-only response
    mock_planner = Mock()
    async def mock_generate(**kwargs):
        return chat_only_response
    mock_planner.generate_workflow = mock_generate
    mock_planner.set_current_workflow = Mock()
    
    # Create test request
    request = WorkflowRequest(
        query="what tools do you have available?",
        refine=False
    )
    
    # Mock current_workflow that might be a WorkflowSpecLLM from previous chat
    mock_current_workflow = WorkflowSpecLLM(
        title="Previous Chat",
        description="Previous chat response",
        reasoning="Previous reasoning",
        nodes=None,
        edges=None
    )
    
    print("🧪 Testing chat-only response handling...")
    
    # Test with WorkflowSpecLLM as current_workflow (should not error)
    with patch('iointel.src.web.workflow_server.current_workflow', mock_current_workflow):
        with patch('iointel.src.web.workflow_server.planner', mock_planner):
            with patch('iointel.src.web.workflow_server.tool_catalog', {}):
                try:
                    response = await generate_workflow(request)
                    print("✅ No error accessing .rev on chat-only response")
                    print(f"   Response success: {response.success}")
                    print(f"   Response has workflow: {response.workflow is not None}")
                    print(f"   Agent response: {response.agent_response[:50]}..." if response.agent_response else "No agent response")
                except AttributeError as e:
                    if "rev" in str(e):
                        print(f"❌ ERROR: {e}")
                        raise
    
    # Test with normal WorkflowSpec as current_workflow (should work)
    import uuid
    normal_workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Normal Workflow",
        description="A normal workflow",
        reasoning="Normal reasoning",
        nodes=[],
        edges=[]
    )
    
    with patch('iointel.src.web.workflow_server.current_workflow', normal_workflow):
        with patch('iointel.src.web.workflow_server.planner', mock_planner):
            with patch('iointel.src.web.workflow_server.tool_catalog', {}):
                try:
                    response = await generate_workflow(request)
                    print("✅ No error with normal WorkflowSpec as current_workflow")
                except AttributeError as e:
                    if "rev" in str(e):
                        print(f"❌ ERROR: {e}")
                        raise
    
    print("\n✅ All tests passed! Chat-only responses are handled correctly.")


if __name__ == "__main__":
    asyncio.run(test_chat_only_response())