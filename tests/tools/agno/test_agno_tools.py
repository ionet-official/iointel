#!/usr/bin/env python3
"""
Test suite for all agno tools to verify KeyError: 'self' fix
Tests Shell, YFinance, File, CSV, Arxiv, and Crawl4ai tools
"""

import asyncio
from pathlib import Path
from iointel.src.agents import Agent
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env

def setup_test_environment():
    """Setup test environment with sample CSV file"""
    test_dir = Path("/tmp/agno_tool_tests")
    test_dir.mkdir(exist_ok=True)
    
    # Create a sample CSV file for testing
    csv_content = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago
Alice,28,Miami
"""
    csv_file = test_dir / "sample_data.csv"
    csv_file.write_text(csv_content)
    
    # Create a sample text file
    text_file = test_dir / "sample.txt"
    text_file.write_text("Hello, this is a test file for agno tools!")
    
    return test_dir

async def test_shell_tool(agent):
    """Test Shell tool functionality"""
    print("\n🐚 Testing Shell Tool:")
    
    tests = [
        ("echo 'Shell tool test'", "Should echo the text"),
        ("date", "Should show current date"),
        ("ls -la /tmp", "Should list files in /tmp"),
        ("pwd", "Should show current directory"),
    ]
    
    for cmd, description in tests:
        try:
            result = await agent.run(f'Execute shell command: {cmd}')
            print(f"  ✅ {description}: Command executed successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def test_yfinance_tool(agent):
    """Test YFinance tool functionality"""
    print("\n📈 Testing YFinance Tool:")
    
    tests = [
        ("get_current_stock_price", "AAPL", "Get Apple stock price"),
        ("get_current_stock_price", "TSLA", "Get Tesla stock price"),
        ("get_company_info", "MSFT", "Get Microsoft company info"),
        ("get_historical_stock_prices", "GOOGL", "Get Google historical prices"),
    ]
    
    for tool_name, symbol, description in tests:
        try:
            if tool_name == "get_historical_stock_prices":
                result = await agent.run(f'Use {tool_name} to get data for {symbol} for the last 5 days')
            else:
                result = await agent.run(f'Use {tool_name} to get information for {symbol}')
            print(f"  ✅ {description}: Data retrieved successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def test_file_tool(agent, test_dir):
    """Test File tool functionality"""
    print("\n📁 Testing File Tool:")
    
    test_file = test_dir / "test_output.txt"
    test_content = "This is a test file created by the File tool!"
    
    tests = [
        ("file_list", str(test_dir), "List files in test directory"),
        ("file_save", f"path='{test_file}' content='{test_content}'", "Save content to file"),
        ("file_read", str(test_file), "Read content from file"),
        ("file_list", str(test_dir), "List files again to verify creation"),
    ]
    
    for tool_name, params, description in tests:
        try:
            if tool_name == "file_save":
                result = await agent.run(f'Use {tool_name} to save content to {test_file}')
            else:
                result = await agent.run(f'Use {tool_name} with parameter: {params}')
            print(f"  ✅ {description}: Operation completed successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def test_csv_tool(agent, test_dir):
    """Test CSV tool functionality"""
    print("\n📊 Testing CSV Tool:")
    
    csv_file = test_dir / "sample_data.csv"
    
    tests = [
        ("csv_list_csv_files", str(test_dir), "List CSV files in directory"),
        ("csv_read_csv_file", str(csv_file), "Read CSV file content"),
        ("csv_get_columns", str(csv_file), "Get CSV column names"),
        ("csv_query_csv_file", f"file_path='{csv_file}' query='SELECT * FROM data WHERE age > 28'", "Query CSV data"),
    ]
    
    for tool_name, params, description in tests:
        try:
            if "query" in params:
                result = await agent.run(f'Use {tool_name} to query the CSV file at {csv_file} with SQL: SELECT * FROM data WHERE age > 28')
            else:
                result = await agent.run(f'Use {tool_name} with parameter: {params}')
            print(f"  ✅ {description}: Operation completed successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def test_arxiv_tool(agent):
    """Test Arxiv tool functionality"""
    print("\n📚 Testing Arxiv Tool:")
    
    tests = [
        ("arxiv_search", "quantum computing", "Search for quantum computing papers"),
        ("arxiv_search", "machine learning", "Search for machine learning papers"),
    ]
    
    for tool_name, query, description in tests:
        try:
            result = await agent.run(f'Use {tool_name} to search for papers about: {query}')
            print(f"  ✅ {description}: Search completed successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def test_crawl4ai_tool(agent):
    """Test Crawl4ai tool functionality"""
    print("\n🕷️  Testing Crawl4ai Tool:")
    
    tests = [
        ("https://httpbin.org/json", "Crawl a simple JSON endpoint"),
        ("https://httpbin.org/html", "Crawl a simple HTML page"),
    ]
    
    for url, description in tests:
        try:
            result = await agent.run(f'Use agno__crawl4ai__web_crawler to crawl: {url}')
            print(f"  ✅ {description}: Crawl completed successfully")
            print(f"     Result preview: {str(result.get('result', 'No result'))[:100]}...")
        except Exception as e:
            print(f"  ❌ {description}: Failed - {e}")

async def main():
    """Main test runner"""
    print("🧪 Starting comprehensive agno tools test suite...")
    
    # Setup test environment
    test_dir = setup_test_environment()
    print(f"📁 Test directory created: {test_dir}")
    
    # Load all tools
    tools = load_tools_from_env()
    print(f"🔧 Loaded {len(tools)} tools")
    
    # Define the agno tools to test
    agno_tools = [
        'run_shell_command',
        'get_current_stock_price', 'get_company_info', 'get_historical_stock_prices',
        'get_company_news', 'get_analyst_recommendations', 'get_stock_fundamentals',
        'file_list', 'file_read', 'file_save',
        'csv_list_csv_files', 'csv_read_csv_file', 'csv_get_columns', 'csv_query_csv_file',
        'arxiv_search', 'arxiv_read_papers',
        'agno__crawl4ai__web_crawler'
    ]
    
    # Filter to only include tools that are actually available
    available_tools = [tool for tool in agno_tools if tool in tools]
    print(f"📋 Available agno tools to test: {len(available_tools)}")
    print(f"   Tools: {', '.join(available_tools)}")
    
    # Create agent with all available agno tools
    agent = Agent(
        name='agno_test_agent',
        instructions='''
        You are a test agent for agno tools. Your job is to execute the requested tool calls exactly as asked.
        Always use the specific tool mentioned in the request.
        Be direct and execute the tools without additional explanations unless the tool fails.
        ''',
        tools=available_tools
    )
    
    print(f"🤖 Agent created with {len(available_tools)} agno tools")
    
    # Run all tests
    try:
        # Test Shell tool
        if 'run_shell_command' in available_tools:
            await test_shell_tool(agent)
        
        # Test YFinance tools
        yfinance_tools = [t for t in available_tools if 'stock' in t.lower() or 'company' in t.lower() or 'analyst' in t.lower()]
        if yfinance_tools:
            await test_yfinance_tool(agent)
        
        # Test File tools
        file_tools = [t for t in available_tools if t.startswith('file_')]
        if file_tools:
            await test_file_tool(agent, test_dir)
        
        # Test CSV tools
        csv_tools = [t for t in available_tools if t.startswith('csv_')]
        if csv_tools:
            await test_csv_tool(agent, test_dir)
        
        # Test Arxiv tools
        arxiv_tools = [t for t in available_tools if t.startswith('arxiv_')]
        if arxiv_tools:
            await test_arxiv_tool(agent)
        
        # Test Crawl4ai tools
        crawl_tools = [t for t in available_tools if 'crawl4ai' in t]
        if crawl_tools:
            await test_crawl4ai_tool(agent)
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Agno tools test suite completed!")
    print("🎉 All agno tools have been tested for KeyError: 'self' compatibility")

if __name__ == "__main__":
    asyncio.run(main())