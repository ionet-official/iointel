"""
Test conversation memory functionality for WorkflowPlanner.
This test verifies that the conversation memory fix works correctly.
"""

import pytest
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

from iointel.src.agent_methods.agents.workflow_agent import WorkflowPlanner
from iointel import AsyncMemory

# Load environment variables
env_path = Path(__file__).parent.parent / "creds.env"
load_dotenv(env_path)


@pytest.fixture
def mock_tool_catalog():
    """Simple tool catalog for testing."""
    return {
        "weather_api": {
            "name": "weather_api",
            "description": "Get weather information for a location", 
            "parameters": {"location": "string", "units": "string"},
            "returns": ["weather_data", "status"]
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {"to": "string", "subject": "string", "body": "string"},
            "returns": ["sent", "delivery_id"]
        }
    }


class TestConversationMemory:
    """Test conversation memory functionality."""

    @pytest.mark.asyncio
    async def test_conversation_memory_workflow_planner(self, mock_tool_catalog):
        """Test that WorkflowPlanner conversation memory works correctly with structured output."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        print("\n=== Testing WorkflowPlanner Conversation Memory ===")
        
        # Create memory instance
        memory = AsyncMemory("sqlite+aiosqlite:///test_conversation_memory.db")
        await memory.init_models()
        
        # Create unique conversation ID for this test
        conversation_id = f"test_workflow_memory_{uuid.uuid4().hex[:8]}"
        
        # Create WorkflowPlanner with memory
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            memory=memory,
            conversation_id=conversation_id,
            debug=True
        )
        
        print(f"Using conversation ID: {conversation_id}")
        
        # Test 1: Generate first workflow
        print("\n1. Generating first workflow...")
        result1 = await planner.generate_workflow(
            query="create a simple hello world workflow with a data source and agent",
            tool_catalog=mock_tool_catalog
        )
        
        assert result1 is not None
        assert result1.title is not None
        print(f"✅ First workflow: {result1.title}")
        
        # Test 2: Generate second workflow (should remember the first)
        print("\n2. Generating second workflow (should remember first)...")
        result2 = await planner.generate_workflow(
            query="now add a decision node to that workflow",
            tool_catalog=mock_tool_catalog
        )
        
        assert result2 is not None
        assert result2.title is not None
        print(f"✅ Second workflow: {result2.title}")
        
        # Test 3: Check conversation memory
        print("\n3. Checking conversation memory...")
        messages = await memory.get_message_history(conversation_id, 20)
        
        print(f"Retrieved {len(messages) if messages else 0} messages from conversation")
        
        # Verify memory is working
        assert messages is not None, "Should be able to retrieve messages"
        assert len(messages) > 0, "Should have messages in conversation"
        
        # Check for structured output messages (tool-call/tool-return)
        has_tool_calls = False
        has_tool_returns = False
        workflow_generations = 0
        
        for msg in messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    part_kind = getattr(part, 'part_kind', 'unknown')
                    if part_kind == 'tool-call':
                        has_tool_calls = True
                        tool_name = getattr(part, 'tool_name', '')
                        if 'final_result' in tool_name:
                            workflow_generations += 1
                    elif part_kind == 'tool-return':
                        has_tool_returns = True
        
        print(f"Found {workflow_generations} workflow generations in memory")
        print(f"Has tool-call messages: {has_tool_calls}")
        print(f"Has tool-return messages: {has_tool_returns}")
        
        # Verify structured output is preserved
        assert has_tool_calls, "Should have tool-call messages (structured output)"
        assert has_tool_returns, "Should have tool-return messages (structured output)"
        assert workflow_generations >= 2, f"Should have at least 2 workflow generations, found {workflow_generations}"
        
        # Test 4: Verify memory persistence across new planner instance
        print("\n4. Testing memory persistence with new planner instance...")
        new_planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            memory=memory,
            conversation_id=conversation_id,
            debug=True
        )
        
        # Generate third workflow (should remember previous two)
        result3 = await new_planner.generate_workflow(
            query="create a summary of all the workflows we've made so far",
            tool_catalog=mock_tool_catalog
        )
        
        assert result3 is not None
        print(f"✅ Third workflow (with memory): {result3.title}")
        
        # Final memory check
        final_messages = await memory.get_message_history(conversation_id, 30)
        print(f"Final conversation has {len(final_messages) if final_messages else 0} messages")
        
        assert len(final_messages) > len(messages), "Should have more messages after third generation"
        
        print("\n✅ CONVERSATION MEMORY TEST PASSED!")
        print("   - WorkflowPlanner can generate multiple workflows")
        print("   - Memory system preserves structured output (tool-call/tool-return)")
        print("   - Memory persists across planner instances")
        print("   - Conversation history is maintained correctly")
        
        # Clean up test database (AsyncMemory doesn't need explicit close)
        # await memory.close()  # Not needed for AsyncMemory

    @pytest.mark.asyncio
    async def test_memory_filtering_fix(self, mock_tool_catalog):
        """Test that the memory filtering fix works for structured output conversations."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        print("\n=== Testing Memory Filtering Fix ===")
        
        # Create memory instance
        memory = AsyncMemory("sqlite+aiosqlite:///test_memory_filtering.db")
        await memory.init_models()
        
        # Test with workflow planner conversation ID pattern
        conversation_id = f"workflow_planner_{uuid.uuid4().hex[:8]}"
        
        # Create WorkflowPlanner with memory
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            memory=memory,
            conversation_id=conversation_id,
            debug=True
        )
        
        print(f"Using workflow planner conversation ID: {conversation_id}")
        
        # Generate a workflow
        result = await planner.generate_workflow(
            query="create a simple workflow",
            tool_catalog=mock_tool_catalog
        )
        
        assert result is not None
        print(f"✅ Generated workflow: {result.title}")
        
        # Check that memory contains structured output messages
        messages = await memory.get_message_history(conversation_id, 10)
        
        assert messages is not None
        assert len(messages) > 0
        
        # Verify that tool-call and tool-return messages are preserved
        tool_call_count = 0
        tool_return_count = 0
        
        for msg in messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    part_kind = getattr(part, 'part_kind', 'unknown')
                    if part_kind == 'tool-call':
                        tool_call_count += 1
                    elif part_kind == 'tool-return':
                        tool_return_count += 1
        
        print(f"Found {tool_call_count} tool-call messages and {tool_return_count} tool-return messages")
        
        # The fix should preserve these messages for workflow planner conversations
        assert tool_call_count > 0, "Should have tool-call messages (structured output)"
        assert tool_return_count > 0, "Should have tool-return messages (structured output)"
        
        print("✅ Memory filtering fix is working correctly!")
        print("   - Structured output messages are preserved for workflow planner conversations")
        print("   - Memory system correctly identifies structured output conversations")
        
        # Clean up (AsyncMemory doesn't need explicit close)
        # await memory.close()  # Not needed for AsyncMemory


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
