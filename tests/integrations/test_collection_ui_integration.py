#!/usr/bin/env python3
"""
Test Collection UI Integration
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from iointel.src.agent_methods.tools.user_input import user_input, prompt_tool
from iointel.src.agent_methods.tools.collection_manager import create_collection
from iointel.src.web.ui_components.text_input_ui import get_text_input_ui_config


async def test_collection_ui_integration():
    """Test the collection UI integration."""
    print("🧪 Testing Collection UI Integration")
    print("=" * 50)
    
    # Test 1: Create a collection with sample data
    print("\n1. Creating test collection:")
    
    collection_result = create_collection(
        name="UI Test Collection",
        records=[
            "What is the capital of France?",
            "How do I learn Python programming?",
            "What are the benefits of meditation?",
            "How do I improve my productivity?",
            "What is machine learning?"
        ],
        description="Test collection for UI integration",
        tags=["test", "questions"]
    )
    
    collection_id = collection_result.get('collection_id')
    print(f"✅ Created collection: {collection_result['collection_name']}")
    print(f"   Collection ID: {collection_id}")
    
    # Test 2: Test user_input with collection loading
    print("\n2. Testing user_input with collection suggestions:")
    
    user_input_result = user_input(
        prompt="What would you like to know?",
        collection_id=collection_id,
        load_suggestions=True,
        save_to_collection="User Questions"
    )
    
    print(f"✅ User input created with {len(user_input_result.get('suggestions', []))} suggestions")
    print(f"   Collection data: {user_input_result.get('collection_data', {}).get('name', 'None')}")
    print(f"   Tool type: {user_input_result.get('tool_type')}")
    print(f"   Status: {user_input_result.get('status')}")
    
    # Test 3: Test UI config for collections
    print("\n3. Testing UI configuration for collection support:")
    
    for tool_name in ["user_input", "prompt_tool"]:
        config = get_text_input_ui_config(tool_name)
        if config:
            print(f"✅ {tool_name} config:")
            print(f"   Supports collections: {config.get('supports_collections', False)}")
            print(f"   Show suggestions: {config.get('show_suggestions', False)}")
            print(f"   Show collection actions: {config.get('show_collection_actions', False)}")
            print(f"   Max suggestions: {config.get('max_suggestions', 5)}")
    
    # Test 4: Test prompt_tool with collection support
    print("\n4. Testing prompt_tool with collection support:")
    
    prompt_result = prompt_tool(
        message="This message will be overridden by collection",
        collection_id=collection_id,
        save_to_collection="System Prompts"
    )
    
    print(f"✅ Prompt tool result: {prompt_result['message']}")
    print(f"   Tool type: {prompt_result['tool_type']}")
    print(f"   Status: {prompt_result['status']}")
    
    # Test 5: Test user_input without collection (fallback)
    print("\n5. Testing user_input without collection (fallback):")
    
    fallback_result = user_input(
        prompt="Enter your favorite color",
        load_suggestions=False
    )
    
    print("✅ Fallback user input created")
    print(f"   Suggestions: {len(fallback_result.get('suggestions', []))}")
    print(f"   Tool type: {fallback_result.get('tool_type')}")
    print(f"   Status: {fallback_result.get('status')}")
    
    # Test 6: Test UI JSON serialization
    print("\n6. Testing UI JSON serialization:")
    
    import json
    
    # Test serializing user_input result with suggestions
    try:
        serialized = json.dumps(user_input_result, default=str)
        print(f"✅ User input result serialized successfully ({len(serialized)} chars)")
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
    
    # Test 7: Show example frontend integration
    print("\n7. Example frontend integration:")
    
    if user_input_result.get('suggestions'):
        print("   Frontend would receive:")
        print(f"   - inputData.suggestions: {user_input_result['suggestions'][:2]}...")
        print(f"   - inputData.collection_data: {user_input_result.get('collection_data', {}).get('name', 'None')}")
        print(f"   - inputData.tool_name: {user_input_result.get('tool_type')}")
        print("\n   JavaScript would call:")
        print("   createEnhancedTextInput(nodeId, toolName, inputData, defaultValue, suggestions, collectionData)")
    
    print("\n✅ All collection UI integration tests completed!")
    return True


if __name__ == "__main__":
    asyncio.run(test_collection_ui_integration())