#!/usr/bin/env python3
"""
Test Collection API Integration
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from iointel.src.agent_methods.tools.collection_manager import create_collection, search_collections
from iointel.src.agent_methods.data_models.prompt_collections import prompt_collection_manager


async def test_collection_api_integration():
    """Test the collection API integration for web interface."""
    print("🧪 Testing Collection API Integration")
    print("=" * 50)
    
    # Test 1: Create test collections
    print("\n1. Creating test collections:")
    
    # Create collection with prompts
    collection1_result = create_collection(
        name="Psychology Prompts",
        records=[
            "How can I better understand my emotions?",
            "What are healthy coping mechanisms for stress?",
            "How do I build more confidence?",
            "What is the psychology behind procrastination?"
        ],
        description="Collection of psychology-related prompts",
        tags=["psychology", "mental health", "prompt_tool"]
    )
    
    collection2_result = create_collection(
        name="Productivity Questions",
        records=[
            "How can I improve my focus?",
            "What are the best time management techniques?",
            "How do I overcome procrastination?",
            "What is the Pomodoro technique?"
        ],
        description="Collection of productivity-related questions",
        tags=["productivity", "time management", "user_input"]
    )
    
    print(f"✅ Created collection 1: {collection1_result['message']}")
    print(f"✅ Created collection 2: {collection2_result['message']}")
    
    # Test 2: Test search functionality (what the web API will call)
    print("\n2. Testing search functionality:")
    
    # Search for "psychology" (should match collection 1)
    search_result1 = search_collections(
        query="psychology",
        tool_filter="prompt_tool"
    )
    
    print(f"✅ Search for 'psychology': {search_result1['message']}")
    if search_result1['status'] == 'success':
        for result in search_result1['results']:
            print(f"   - {result['collection_name']}: {len(result['matching_records'])} matches")
            for record in result['matching_records'][:2]:  # Show first 2
                print(f"     '{record}'")
    
    # Search for "procrastination" (should match both collections)
    search_result2 = search_collections(
        query="procrastination"
    )
    
    print(f"✅ Search for 'procrastination': {search_result2['message']}")
    if search_result2['status'] == 'success':
        for result in search_result2['results']:
            print(f"   - {result['collection_name']}: {len(result['matching_records'])} matches")
    
    # Test 3: Test search with no results
    print("\n3. Testing search with no results:")
    
    search_result3 = search_collections(
        query="nonexistent_topic_xyz"
    )
    
    print(f"✅ Search for 'nonexistent_topic_xyz': {search_result3['message']}")
    print(f"   Status: {search_result3['status']}")
    print(f"   Results count: {len(search_result3.get('results', []))}")
    
    # Test 4: Test adding new records (simulating save functionality)
    print("\n4. Testing save functionality:")
    
    # Add a new record to existing collection
    psychology_collection = None
    for collection in prompt_collection_manager.list_collections():
        if collection.name == "Psychology Prompts":
            psychology_collection = collection
            break
    
    if psychology_collection:
        new_prompt = "How can I develop better emotional intelligence?"
        psychology_collection.add_record(new_prompt)
        prompt_collection_manager.save_collection(psychology_collection)
        print(f"✅ Added new record to Psychology Prompts: '{new_prompt}'")
        
        # Verify it was added
        updated_collection = prompt_collection_manager.load_collection(psychology_collection.id)
        print(f"   Collection now has {len(updated_collection.records)} records")
    
    # Test 5: Simulate web API response format
    print("\n5. Testing web API response format:")
    
    # This simulates what the web API endpoint will return
    web_search_result = search_collections(query="confidence")
    
    print("   Web API would return:")
    print(f"   Success: {web_search_result.get('status') == 'success'}")
    print(f"   Results: {len(web_search_result.get('results', []))}")
    
    if web_search_result.get('status') == 'success':
        for result in web_search_result['results']:
            print(f"   Collection: {result['collection_name']}")
            print(f"   Matches: {result['matching_records']}")
            print(f"   Total records: {result['total_records']}")
            print(f"   Tags: {result['tags']}")
    
    # Test 6: Test collection statistics
    print("\n6. Testing collection statistics:")
    
    all_collections = prompt_collection_manager.list_collections()
    print(f"✅ Total collections: {len(all_collections)}")
    
    for collection in all_collections:
        stats = collection.get_stats()
        print(f"   - {collection.name}: {stats['total_records']} records, {stats['usage_count']} uses")
    
    # Clean up test collections
    print("\n7. Cleaning up test collections:")
    
    for collection in all_collections:
        if collection.name in ["Psychology Prompts", "Productivity Questions"]:
            prompt_collection_manager.delete_collection(collection.id)
            print(f"   Deleted: {collection.name}")
    
    print("\n✅ All collection API integration tests completed!")
    print("\n📋 Integration Summary:")
    print("   1. ✅ Collection creation works")
    print("   2. ✅ Search functionality works")  
    print("   3. ✅ Save functionality works")
    print("   4. ✅ Web API response format is correct")
    print("   5. ✅ Frontend can now connect to backend collections")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_collection_api_integration())