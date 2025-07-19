#!/usr/bin/env python3
"""
Test Prompt Collections System
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from iointel.src.agent_methods.tools.user_input import user_input, prompt_tool
from iointel.src.agent_methods.tools.collection_manager import (
    create_collection, list_collections, get_collection, search_collections,
    add_to_collection, delete_collection, get_popular_records
)
from iointel.src.agent_methods.data_models.prompt_collections import (
    prompt_collection_manager, ListRecords
)


async def test_prompt_collections():
    """Test the prompt collections system."""
    print("🧪 Testing Prompt Collections System")
    print("=" * 50)
    
    # Test 1: Create a collection using the manager directly
    print("\n1. Testing direct collection creation:")
    
    test_prompts = [
        "What is the weather like today?",
        "How do I install Python packages?",
        "Explain machine learning in simple terms",
        "What are the benefits of exercise?",
        "How do I cook pasta?"
    ]
    
    collection = prompt_collection_manager.create_collection_from_records(
        name="Test Queries",
        records=test_prompts,
        description="A collection of test queries for testing",
        tags=["test", "general"]
    )
    
    print(f"✅ Created collection: {collection.name} ({collection.id})")
    print(f"   Records: {len(collection.records)}")
    print(f"   Tags: {collection.tags}")
    
    # Test 2: Test collection manager tool
    print("\n2. Testing collection manager tool:")
    
    # Create collection via tool
    result = create_collection(
        name="Code Questions",
        records=[
            "How do I debug Python code?",
            "What is the difference between list and tuple?",
            "How do I handle exceptions in Python?",
            "What are Python decorators?"
        ],
        description="Programming related questions",
        tags=["programming", "python"]
    )
    
    print(f"✅ Tool result: {result['message']}")
    print(f"   Collection ID: {result.get('collection_id')}")
    
    # Test 3: List collections
    print("\n3. Testing list collections:")
    
    list_result = list_collections()
    print(f"✅ {list_result['message']}")
    
    for collection_info in list_result['collections']:
        print(f"   - {collection_info['name']}: {collection_info['records_count']} records")
    
    # Test 4: Search collections
    print("\n4. Testing search functionality:")
    
    search_result = search_collections("Python")
    print(f"✅ {search_result['message']}")
    
    for result in search_result['results']:
        print(f"   - {result['collection_name']}: {len(result['matching_records'])} matches")
        for record in result['matching_records'][:2]:  # Show first 2
            print(f"     '{record}'")
    
    # Test 5: Test user_input with collection support
    print("\n5. Testing user_input with collection support:")
    
    # Test loading suggestions from collection
    user_input_result = user_input(
        prompt="What programming question do you have?",
        collection_id=result.get('collection_id'),
        save_to_collection="User Questions",
        load_suggestions=True
    )
    
    print(f"✅ User input form created with suggestions")
    print(f"   Suggestions: {len(user_input_result.get('suggestions', []))}")
    print(f"   Collection data: {user_input_result.get('collection_data', {}).get('name', 'None')}")
    
    # Test 6: Test prompt_tool with collection support
    print("\n6. Testing prompt_tool with collection support:")
    
    # Save a prompt to collection
    prompt_result = prompt_tool(
        message="Please analyze the following code for potential improvements",
        save_to_collection="Analysis Prompts"
    )
    
    print(f"✅ Prompt tool executed: {prompt_result['message']}")
    
    # Test loading from collection
    prompt_result_2 = prompt_tool(
        message="This will be overridden",
        collection_id=collection.id  # Use the first collection we created
    )
    
    print(f"✅ Prompt loaded from collection: {prompt_result_2['message']}")
    
    # Test 7: Test popular records
    print("\n7. Testing popular records:")
    
    popular_result = get_popular_records(limit=5)
    print(f"✅ {popular_result['message']}")
    
    for record in popular_result['records']:
        print(f"   - '{record['record']}' (from {record['collection_name']})")
    
    # Test 8: Test collection stats
    print("\n8. Testing collection statistics:")
    
    for collection_info in list_result['collections']:
        collection_detail = get_collection(collection_info['id'])
        if collection_detail['status'] == 'success':
            stats = collection_detail['collection']['stats']
            print(f"   - {collection_info['name']}: {stats['total_records']} records, " +
                  f"{stats['usage_count']} uses, {stats['age_days']} days old")
    
    # Test 9: Simulate user input with collection save
    print("\n9. Testing user input collection save simulation:")
    
    # Simulate user providing input that should be saved to collection
    simulated_user_input = user_input(
        prompt="What is your favorite programming language?",
        save_to_collection="User Preferences",
        execution_metadata={
            'user_inputs': {'test_input': 'Python - I love its simplicity and power'},
            'node_id': 'test_node'
        }
    )
    
    print(f"✅ User input processed: {simulated_user_input['message']}")
    
    # Verify the collection was created/updated
    final_list = list_collections()
    print(f"✅ Final collection count: {final_list['total_count']}")
    
    # Test 10: Clean up (optional - comment out if you want to keep test data)
    print("\n10. Cleaning up test collections:")
    
    for collection_info in list_collections()['collections']:
        if 'test' in collection_info['name'].lower() or 'code' in collection_info['name'].lower():
            delete_result = delete_collection(collection_info['id'])
            print(f"   - Deleted: {collection_info['name']}")
    
    print("\n✅ All prompt collection tests completed successfully!")
    return True


if __name__ == "__main__":
    asyncio.run(test_prompt_collections())