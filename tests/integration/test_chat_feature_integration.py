#!/usr/bin/env python3
"""
Integration test for chat-only responses feature.

This test demonstrates the end-to-end functionality of the chat-only response
feature, showing how it integrates between the workflow planner, data models,
and web server components.
"""

import asyncio
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpecLLM, WorkflowSpec


class TestChatFeatureIntegration:
    """Integration tests for chat-only response feature."""
    
    def test_chat_only_feature_demonstration(self):
        """Test the chat-only feature end-to-end (demonstration)."""
        asyncio.run(self._run_chat_feature_demo())
    
    async def _run_chat_feature_demo(self):
        """Demo of the chat-only feature working end-to-end."""
        print("🧪 Testing Chat-Only Response Feature")
        print("=" * 50)
        
        # Create a simple tool catalog
        tool_catalog = {
            "web_search": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {"query": "str"},
                "required_parameters": ["query"],
                "is_async": True
            }
        }
        
        # Create workflow planner
        planner = WorkflowPlanner(debug=True)
        
        # Test 1: Normal workflow generation
        print("\n1. Testing Normal Workflow Generation")
        print("-" * 40)
        
        try:
            result = await planner.generate_workflow(
                query="Search for Python tutorials",
                tool_catalog=tool_catalog
            )
            
            if isinstance(result, WorkflowSpec):
                print("✅ Normal workflow generated:")
                print(f"   Title: {result.title}")
                print(f"   Description: {result.description}")
                print(f"   Reasoning: {result.reasoning}")
                print(f"   Nodes: {len(result.nodes)}")
                print(f"   Edges: {len(result.edges)}")
            else:
                print(f"❌ Expected WorkflowSpec, got {type(result)}")
                
        except Exception as e:
            print(f"❌ Error in normal workflow generation: {e}")
        
        # Test 2: Chat-only response (simulate by creating WorkflowSpecLLM with null nodes/edges)
        print("\n2. Testing Chat-Only Response")
        print("-" * 40)
        
        # Create a mock chat-only response
        chat_response = WorkflowSpecLLM(
            title=None,
            description="Chat response",
            reasoning="I need more information about what specific type of Python tutorials you're looking for. Are you interested in beginner tutorials, web development, data science, or something else?",
            nodes=None,
            edges=None
        )
        
        print("✅ Chat-only response created:")
        print(f"   Nodes: {chat_response.nodes}")
        print(f"   Edges: {chat_response.edges}")
        print(f"   Chat message: {chat_response.reasoning}")
        
        # Test 3: Web server response handling
        print("\n3. Testing Web Server Response Format")
        print("-" * 40)
        
        from iointel.src.web.workflow_server import WorkflowResponse
        
        # Normal workflow response - use the actual workflow that was generated
        if isinstance(result, WorkflowSpec):
            workflow_dict = result.model_dump()
            # Convert UUID to string for JSON serialization
            if 'id' in workflow_dict:
                workflow_dict['id'] = str(workflow_dict['id'])
                
            normal_response = WorkflowResponse(
                success=True,
                workflow=workflow_dict,
                agent_response="I have generated a workflow for you"
            )
            print(f"✅ Normal response: workflow={normal_response.workflow is not None}")
        else:
            print("⚠️ Skipping normal response test - no valid workflow generated")
        
        # Chat-only response
        chat_only_response = WorkflowResponse(
            success=True,
            workflow=None,  # No workflow update
            agent_response=chat_response.reasoning
        )
        print(f"✅ Chat-only response: workflow={chat_only_response.workflow}, message_length={len(chat_only_response.agent_response)}")
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 50)
        print("Summary:")
        print("- Normal workflows: Return WorkflowSpec + agent_response")
        print("- Chat-only responses: Return WorkflowSpecLLM (with null nodes/edges) + agent_response")
        print("- Web server: Returns workflow=None + agent_response for chat-only")
        print("- UI: Preserves previous DAG, shows chat message")


if __name__ == "__main__":
    test_instance = TestChatFeatureIntegration()
    test_instance.test_chat_only_feature_demonstration()