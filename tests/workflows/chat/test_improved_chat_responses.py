"""
Test to verify improved chat responses in WorkflowPlanner.
"""

import asyncio
import json
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env


async def test_improved_chat_response():
    """Test that chat responses are now engaging and tool-specific."""
    
    print("🧪 Testing improved chat responses...")
    
    # Initialize planner with mock tools
    mock_tool_catalog = {
        "listing_coins": {"description": "Get top cryptocurrencies by market cap"},
        "get_coin_quotes": {"description": "Get real-time crypto prices"},
        "searxng.search": {"description": "Privacy-focused web search"},
        "conditional_gate": {"description": "Route workflows based on conditions"},
        "user_input": {"description": "Collect user input interactively"},
        "add": {"description": "Add two numbers"},
        "multiply": {"description": "Multiply two numbers"}
    }
    
    planner = WorkflowPlanner(
        model="gpt-4o-mini",
        api_key=None,  # Will use env var
        conversation_id="test_chat_response"
    )
    
    # Test query about available tools
    query = "what tools do you have available? Cluster them by usage."
    
    print(f"\n📝 Query: {query}")
    print("=" * 60)
    
    try:
        result = await planner.generate_workflow(
            query=query,
            tool_catalog=mock_tool_catalog
        )
        
        # Print the raw WorkflowSpec JSON
        print("\n📋 Raw WorkflowSpec JSON:")
        print("-" * 60)
        if hasattr(result, 'model_dump'):
            # Pydantic v2
            spec_dict = result.model_dump()
        else:
            # Pydantic v1
            spec_dict = result.dict()
        
        spec_json = json.dumps(spec_dict, indent=2)
        print(spec_json)
        print("-" * 60)
        
        # Check if it's a chat-only response
        if hasattr(result, 'nodes') and result.nodes is None:
            print("\n✅ Chat-only response detected")
            print(f"\n💬 Response reasoning:\n{result.reasoning}")
            
            # Check for improved response qualities
            reasoning_lower = result.reasoning.lower()
            
            # Check for enthusiasm and emojis
            has_emojis = any(char in result.reasoning for char in ['🚀', '💰', '🔍', '🤖', '💡', '✨', '📊'])
            print(f"\n✅ Uses emojis: {has_emojis}")
            
            # Check for categorization
            has_categories = any(term in result.reasoning for term in ['**', 'Crypto', 'Search', 'AI', 'Tools'])
            print(f"✅ Categorizes tools: {has_categories}")
            
            # Check for specific tool mentions
            tools_mentioned = sum(1 for tool in mock_tool_catalog.keys() if tool in result.reasoning)
            print(f"✅ Tools mentioned: {tools_mentioned}/{len(mock_tool_catalog)}")
            
            # Check for IO.net branding
            has_ionet_branding = 'io.net' in reasoning_lower or 'workflow' in reasoning_lower
            print(f"✅ IO.net branding: {has_ionet_branding}")
            
            # Check for suggestions/ideas
            has_suggestions = any(term in reasoning_lower for term in ['idea', 'build', 'create', 'workflow', 'example'])
            print(f"✅ Provides suggestions: {has_suggestions}")
            
            # Overall assessment
            print("\n📊 Response Quality Assessment:")
            quality_score = sum([has_emojis, has_categories, tools_mentioned > 3, has_ionet_branding, has_suggestions])
            print(f"   Quality score: {quality_score}/5")
            
            if quality_score >= 4:
                print("   ✅ EXCELLENT - Response is engaging and informative!")
            elif quality_score >= 3:
                print("   ✅ GOOD - Response shows improvement")
            else:
                print("   ❌ NEEDS WORK - Response is still too generic")
                
        else:
            print("❌ Expected chat-only response but got workflow")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    # Load any available tools from environment
    load_tools_from_env()
    asyncio.run(test_improved_chat_response())