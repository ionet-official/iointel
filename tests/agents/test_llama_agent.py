#!/usr/bin/env python3
"""
Test script for Agent.run() using Llama model.

Based on the pattern found in testing_agents.ipynb, this script tests a simple
agent execution using the Llama model with environment configuration.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from creds.env
load_dotenv('creds.env')

from iointel import Agent, AsyncMemory

async def test_llama_agent():
    """Test a simple Agent.run() call using Llama model."""
    
    print("🧪 Testing Llama Agent.run()")
    print("=" * 40)
    
    # Create agent using pattern from notebook
    agent = Agent(
        name='TestBot',
        instructions='You are a helpful assistant that provides clear, concise answers.',
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv('IO_API_KEY'),
        base_url=os.getenv('IO_BASE_URL', "https://api.intelligence-dev.io.solutions/api/v1")
    )
    
    print(f"📋 Agent Configuration:")
    print(f"   Name: {agent.name}")
    print(f"   Model: {agent.model}")
    print(f"   Base URL: {agent.base_url}")
    print(f"   API Key: {os.getenv('IO_API_KEY')[:20]}..." if os.getenv('IO_API_KEY') else "   API Key: Not found")
    
    # Test simple conversation
    test_prompt = "What is 2 + 2? Please be brief."
    
    print(f"\n🔄 Running agent with prompt: '{test_prompt}'")
    
    try:
        # Run the agent
        result = await agent.run(test_prompt)
        
        print(f"\n✅ Agent execution successful!")
        print(f"📄 Result type: {type(result)}")
        print(f"📝 Response: {result.get('result', result)}")
        
        # Show full result structure if it's a dict
        if isinstance(result, dict):
            print(f"\n🔍 Full result structure:")
            for key, value in result.items():
                if key == 'full_result':
                    print(f"   {key}: {type(value)}")
                else:
                    print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Agent execution failed: {e}")
        print(f"💥 Error type: {type(e)}")
        import traceback
        print(f"📋 Traceback:")
        traceback.print_exc()
        return False

async def test_llama_with_conversation():
    """Test agent with conversation continuity."""
    
    print(f"\n🔄 Testing conversation continuity...")
    
    # Initialize memory for conversation persistence
    memory = AsyncMemory("sqlite+aiosqlite:///test_conversations.db")
    await memory.init_models()
    
    agent = Agent(
        name='ConversationBot',
        instructions='You are a helpful assistant. Remember our conversation context.',
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=os.getenv('IO_API_KEY'),
        base_url=os.getenv('IO_BASE_URL', "https://api.intelligence-dev.io.solutions/api/v1"),
        memory=memory
    )
    
    conversation_id = "test_llama_conversation_001"
    
    try:
        # First message
        result1 = await agent.run("My name is Alice. What's your name?", conversation_id=conversation_id)
        print(f"   👋 First message: {result1.get('result', result1)}")
        
        # Second message using conversation context
        result2 = await agent.run("What name did I just tell you?", conversation_id=conversation_id)
        print(f"   🤔 Context test: {result2.get('result', result2)}")

        assert 'Alice' in result2.get('result', result2), "Alice should be in the result"
        
        return True
        
    except Exception as e:
        print(f"   ❌ Conversation test failed: {e}")
        return False

async def main():
    """Main test execution function."""
    
    print("🚀 Llama Agent Test Suite")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('IO_API_KEY'):
        print("❌ IO_API_KEY not found in environment")
        print("💡 Make sure creds.env is properly configured")
        return
    
    # Run tests
    test_results = []
    
    # Test 1: Basic Agent.run()
    test_results.append(await test_llama_agent())
    
    # Test 2: Conversation continuity
    test_results.append(await test_llama_with_conversation())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    asyncio.run(main())