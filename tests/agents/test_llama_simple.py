#!/usr/bin/env python3
"""
Simple test script to isolate the Llama + tools issue.
"""
import asyncio
from iointel import Agent
from iointel import AsyncMemory
from rich.console import Console
from dotenv import load_dotenv

# Initialize rich console
console = Console()

# Load environment variables first
load_dotenv("creds.env")

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_llama_simple"

# Test with no tools first
console.print("[blue]Testing Llama agent with NO tools...[/blue]")

# Get model configuration using centralized function
from iointel.src.utilities.constants import get_model_config
model_config = get_model_config(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,  # Let it use defaults from environment
    base_url=None  # Let it use defaults from environment
)

console.print(f"[green]🔧 Using model: {model_config['model']}[/green]")
console.print(f"[blue]🌐 API endpoint: {model_config['base_url']}[/blue]")
console.print(f"[yellow]🔑 API key: {'✓ Set' if model_config['api_key'] else '✗ Missing'}[/yellow]")

try:
    # Create the agent with NO tools
    runner = Agent(
        name="TestLlama",
        instructions="You are a helpful assistant for testing Llama models.",
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        memory=memory,
        conversation_id=CONVERSATION_ID,
        tools=[],  # NO TOOLS
        show_tool_calls=True,
        debug=False,
    )
    console.print("[green]✅ Agent created successfully with no tools[/green]")
    
    # Test basic conversation
    async def test_no_tools():
        console.print("[cyan]Testing basic conversation...[/cyan]")
        result = await runner.run("Hello! Can you tell me what 2+2 equals?", pretty=True)
        console.print("[green]✅ No-tools test successful[/green]")
        return result
    
    # Run the test
    result = asyncio.run(test_no_tools())
    console.print("[green]✅ Basic Llama agent works without tools[/green]")
    
except Exception as e:
    console.print(f"[red]❌ Error creating agent with NO tools: {str(e)}[/red]")
    import traceback
    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
    exit(1)

# Now test with a simple tool
console.print("\n[blue]Testing Llama agent with simple calculator tool...[/blue]")

# Define a simple calculator tool
def simple_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Register the tool
from iointel.src.utilities.decorators import register_tool
register_tool("simple_add")(simple_add)

try:
    # Create agent with the simple tool
    runner_with_tool = Agent(
        name="TestLlamaWithTool",
        instructions="You are a helpful assistant with a calculator tool.",
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        memory=memory,
        conversation_id=CONVERSATION_ID + "_with_tool",
        tools=["simple_add"],  # Simple tool
        show_tool_calls=True,
        debug=False,
    )
    console.print("[green]✅ Agent created successfully with simple tool[/green]")
    
    # Test with tool usage
    async def test_with_tool():
        console.print("[cyan]Testing tool usage...[/cyan]")
        result = await runner_with_tool.run("Please use the simple_add tool to calculate 5 + 7", pretty=True)
        console.print("[green]✅ Tool test successful[/green]")
        return result
    
    # Run the test
    result = asyncio.run(test_with_tool())
    console.print("[green]✅ Llama agent works with simple tools[/green]")
    
except Exception as e:
    console.print(f"[red]❌ Error creating agent with simple tool: {str(e)}[/red]")
    import traceback
    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
    exit(1)

console.print("\n[green]🎉 All tests passed! Llama agent works with both no tools and simple tools.[/green]")