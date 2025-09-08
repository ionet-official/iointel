"""
Test markdown rendering in the web interface.
"""

import asyncio
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env


async def test_markdown_rendering():
    """Test that markdown is properly rendered in chat responses."""
    
    print("🧪 Testing markdown rendering in chat responses...")
    
    # Initialize planner with comprehensive tool catalog
    mock_tool_catalog = {
        "listing_coins": {"description": "Get top cryptocurrencies by market cap"},
        "get_coin_quotes": {"description": "Get real-time crypto prices"},
        "searxng.search": {"description": "Privacy-focused web search"},
        "conditional_gate": {"description": "Route workflows based on conditions"},
        "user_input": {"description": "Collect user input interactively"},
        "add": {"description": "Add two numbers"},
        "multiply": {"description": "Multiply two numbers"},
        "get_company_info": {"description": "Fetch company information"},
        "calculator_advanced": {"description": "Advanced calculator operations"},
        "calculator_exponential": {"description": "Exponential calculations"},
        "calculator_logarithm": {"description": "Logarithm calculations"},
        "calculator_prime": {"description": "Prime number operations"},
        "calculator_square": {"description": "Square operations"},
        "coin_conversion_rates": {"description": "Get crypto conversion rates"},
        "get_coin_historical": {"description": "Get historical crypto data"}
    }
    
    planner = WorkflowPlanner(
        model="gpt-4o-mini",
        api_key=None,  # Will use env var
        conversation_id="test_markdown"
    )
    
    # Test query that should trigger rich markdown response
    query = "what tools do you have available?"
    
    print(f"\n📝 Query: {query}")
    print("=" * 60)
    
    try:
        result = await planner.generate_workflow(
            query=query,
            tool_catalog=mock_tool_catalog
        )
        
        # Check if it's a chat-only response with markdown
        if hasattr(result, 'nodes') and result.nodes is None:
            print("\n✅ Chat-only response detected")
            print(f"\n💬 Response with markdown:\n{'-' * 60}\n{result.reasoning}\n{'-' * 60}")
            
            # Check for markdown elements
            markdown_elements = {
                "Headers": ["# ", "## ", "### "],
                "Bold": ["**"],
                "Lists": ["- ", "* ", "1. "],
                "Code": ["`"],
                "Line breaks": ["\n\n"]
            }
            
            print("\n📊 Markdown Analysis:")
            for element, patterns in markdown_elements.items():
                found = any(pattern in result.reasoning for pattern in patterns)
                print(f"   {element}: {'✅' if found else '❌'}")
            
            # Check specific formatting expectations
            print("\n🎨 Expected Formatting Elements:")
            
            # Should have category headers
            has_category_headers = any(header in result.reasoning for header in 
                ["**🚀", "**💰", "**🔍", "**🤖", "**📊", "## "])
            print(f"   Category headers: {'✅' if has_category_headers else '❌'}")
            
            # Should have tool lists with descriptions
            has_tool_lists = "- " in result.reasoning or "* " in result.reasoning
            print(f"   Tool lists: {'✅' if has_tool_lists else '❌'}")
            
            # Should have emojis for visual appeal
            emojis = ['🚀', '💰', '🔍', '🤖', '💡', '✨', '📊', '🎯', '⚡']
            emoji_count = sum(1 for emoji in emojis if emoji in result.reasoning)
            print(f"   Emojis used: {emoji_count} ({'✅' if emoji_count >= 5 else '❌ Too few'})")
            
            # Should mention IO.net
            has_branding = "IO.net" in result.reasoning or "io.net" in result.reasoning.lower()
            print(f"   IO.net branding: {'✅' if has_branding else '❌'}")
            
            # Print a sample of how it would render
            print("\n🖼️  Preview of rendered output (first 500 chars):")
            print("-" * 60)
            # Simulate basic markdown rendering
            preview = result.reasoning[:500]
            preview = preview.replace("**", "")  # Remove bold markers for preview
            preview = preview.replace("### ", "▶ ")  # Show headers
            preview = preview.replace("## ", "▶▶ ")
            preview = preview.replace("# ", "▶▶▶ ")
            print(preview + "...")
            
        else:
            print("❌ Expected chat-only response but got workflow")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Markdown rendering test complete!")
    print("\n💡 To see the actual rendered output:")
    print("   1. Start the workflow server: python -m iointel.src.web.workflow_server")
    print("   2. Open http://localhost:7777 in your browser")
    print("   3. Type: 'what tools do you have available?'")
    print("   4. Observe the markdown formatting with:")
    print("      - Bold category headers")
    print("      - Organized tool lists")
    print("      - Emojis for visual appeal")
    print("      - Proper spacing and structure")


if __name__ == "__main__":
    # Load any available tools from environment
    load_tools_from_env()
    asyncio.run(test_markdown_rendering())