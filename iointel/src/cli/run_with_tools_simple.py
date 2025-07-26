import asyncio
from iointel import Agent
from iointel import AsyncMemory
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from iointel.src.agent_methods.tools.discovery import load_tools_from_env
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv("creds.env")

# Initialize rich console
console = Console()

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_conversation_test"

# Load all available tools based on environment credentials
console.print("[blue]Loading tools based on available credentials...[/blue]")
available_tools = load_tools_from_env("creds.env")

# Combine all tools
all_tools = available_tools 

console.print(f"[green]Loaded {len(available_tools)} credential-based tools[/green]")
console.print(f"[yellow]Total tools available: {len(all_tools)}[/yellow]")

# Create the agent with tools
runner = Agent(
    name="Solar",
    instructions="""You are a helpful assistant with access to various tools.
    You can perform calculations, check weather information, scrape websites, search the web,
    query financial data, and more depending on what tools are available.
    Always explain your reasoning before using a tool.
    After using a tool, explain the result to the user. You love giving detailed explanations when given the opportunity.""",
    model=os.getenv("MODEL_NAME", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    memory=memory,
    conversation_id=CONVERSATION_ID,
    tools=all_tools,
    show_tool_calls=True,  # Enable verbose output to show tool usage
    tool_pil_layout="horizontal",  # 'vertical' or 'horizontal'
    debug=False,
)


async def main():
    console.print(
        Panel.fit(
            "Starting enhanced chat with tools (type 'exit' to quit)...",
            title="Tool-Enhanced Chat",
            border_style="blue",
        )
    )
    
    # Display available tools
    console.print("\n[cyan]Available tools:[/cyan]")
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    for tool_name in sorted(TOOLS_REGISTRY.keys()):
        tool = TOOLS_REGISTRY[tool_name]
        console.print(f"  • {tool_name}: {tool.description}")
    console.print()

    while True:
        try:
            input_text = input(">>>>: ")
            if input_text.lower() == "exit":
                break

            # Run the agent and capture the result
            await runner.run(
                input_text,
                pretty=True
            )

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            console.print(f"[red]Error type:[/red] {type(e)}")
            import traceback

            console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")


if __name__ == "__main__":
    # Load environment variables from creds.env
    load_dotenv("creds.env")
    asyncio.run(main())