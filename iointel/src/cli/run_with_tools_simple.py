import asyncio
from iointel import Agent
from iointel import AsyncMemory
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from iointel.src.agent_methods.tools.discovery import load_tools_from_env
import os
import logging
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize rich console
console = Console()

# Load environment variables first
load_dotenv("creds.env")

# Load tools using the same pattern as test fixtures
console.print("[blue]Loading tools based on available credentials...[/blue]")
# Step 1: Load tools from environment (this registers them in TOOLS_REGISTRY)
tool_names = load_tools_from_env("creds.env")

# Step 2: Create tool catalog from registered tools
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
tool_catalog = create_tool_catalog()

# Step 3: Filter tools to exclude those that cause pydantic_ai issues
def is_safe_for_pydantic_ai(tool_name: str) -> bool:
    """Check if a tool is safe to use with pydantic_ai Agent."""
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    
    if tool_name not in TOOLS_REGISTRY:
        return False
    
    try:
        tool = TOOLS_REGISTRY[tool_name]
        func = tool.get_wrapped_fn()
        
        # Try to get type hints - if this fails, the tool is problematic
        import typing
        import inspect
        
        sig = inspect.signature(func)
        type_hints = typing.get_type_hints(func)
        
        # Check for bound method issues (self in signature but not in type hints)
        if 'self' in sig.parameters and 'self' not in type_hints:
            return False
            
        return True
    except Exception:
        # If we can't safely inspect, exclude it
        return False

# Filter out problematic tools
safe_tools = [tool_name for tool_name in tool_names if is_safe_for_pydantic_ai(tool_name)]
filtered_count = len(tool_names) - len(safe_tools)

if filtered_count > 0:
    console.print(f"[yellow]Filtered out {filtered_count} tools with pydantic_ai compatibility issues[/yellow]")

available_tools = safe_tools

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_conversation_test"

# Combine all tools
all_tools = available_tools 

console.print(f"[green]Loaded {len(available_tools)} credential-based tools[/green]")
console.print(f"[yellow]Total tools available: {len(all_tools)}[/yellow]")

def select_model():
    """Interactive model selection."""
    console.print("\n[cyan]🤖 Model Selection[/cyan]")
    
    available_models = [
        ("gpt-4o", "GPT-4o (OpenAI) - Most capable, structured output"),
        ("gpt-4o-mini", "GPT-4o Mini (OpenAI) - Fast and efficient"),
        ("meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B (IO Intel) - Open source, great for conversation"),
        ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Maverick-17B (IO Intel) - Compact, efficient"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "Qwen3-Coder-480B-A35B-Instruct-FP8 (IO Intel) - Compact, efficient"),
    ]
    
    for i, (model_name, description) in enumerate(available_models, 1):
        console.print(f"  {i}. {description}")
    
    console.print("  6. Custom (enter your own model name)")
    console.print(f"  7. Use environment MODEL_NAME ({os.getenv('MODEL_NAME', 'not set')})")
    
    
    while True:
        try:
            choice = input("\nSelect model (1-7): ").strip()
            
            # Handle model selections (1-5)
            if choice in ["1", "2", "3", "4", "5"]:
                return available_models[int(choice) - 1][0]
            # Handle custom model (6)
            elif choice == "6":
                custom_model = input("Enter custom model name: ").strip()
                return custom_model if custom_model else "gpt-4o"
            # Handle environment variable (7)
            elif choice == "7":
                return os.getenv("MODEL_NAME", "gpt-4o")
            else:
                console.print("[red]Invalid choice. Please select 1-7.[/red]")
                
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please select 1-7.[/red]")

# Select model interactively
selected_model = select_model()

# Get model configuration using centralized function
from iointel.src.utilities.constants import get_model_config
model_config = get_model_config(
    model=selected_model,
    api_key=None,  # Let it use defaults from environment
    base_url=None  # Let it use defaults from environment
)

console.print(f"\n[green]🔧 Using model: {model_config['model']}[/green]")
console.print(f"[blue]🌐 API endpoint: {model_config['base_url']}[/blue]")
console.print(f"[yellow]🔑 API key: {'✓ Set' if model_config['api_key'] else '✗ Missing'}[/yellow]")

# Create the agent with tools
# Detect if we're using a Llama model and adjust instructions accordingly
is_llama_model = "llama" in model_config["model"].lower()

if is_llama_model:
    console.print("[yellow]🦙 Detected Llama model - using optimized function calling instructions[/yellow]")
    agent_instructions = """You are a helpful assistant with tool calling capabilities. 

Environment: ipython

When you need to use a tool, respond with a JSON function call in this exact format: 

{"name": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}. 
 
 Available functions will be automatically provided to you. Always use the exact    
 function names and parameter names as specified. You can perform calculations, check weather information, scrape websites,    
 search the web, query financial data, and more depending on what tools are available. Always explain your reasoning before    
 using a tool and explain the result after using it. Additionally, after completing a task, update the TODO file by reading    
 its current contents, appending the relevant information, and saving the updated contents. Use the file_read and file_save    
 tools to achieve this.
 Be Creative and Remember to use tools to achieve your goals."""
    print("===========\n", agent_instructions, "\n===========")
else:
    agent_instructions = """You are a helpful assistant with access to various tools.
    You can perform calculations, check weather information, scrape websites, search the web,
    query financial data, and more depending on what tools are available.
    Always explain your reasoning after using a tool.
    After using a tool, explain the result to the user. You love giving detailed explanations when given the opportunity."""

runner = Agent(
    name="Solar",
    instructions=agent_instructions,
    model=model_config["model"],
    api_key=model_config["api_key"],
    base_url=model_config["base_url"],
    memory=memory,
    conversation_id=CONVERSATION_ID+str(datetime.now()),
    tools=all_tools,
    show_tool_calls=True,  # Enable verbose output to show tool usage
    tool_pil_layout="horizontal",  # 'vertical' or 'horizontal'
    debug=False,
)


async def main():
    console.print(
        Panel.fit(
            f"🚀 Enhanced Chat with Tools\n\n"
            f"Model: {model_config['model']}\n"
            f"API: {model_config['base_url']}\n"
            f"Tools: {len(all_tools)} available\n\n"
            f"Type 'exit' to quit, 'tools' to list tools, 'model' to show model info",
            title="Tool-Enhanced Agent Chat",
            border_style="green",
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
            elif input_text.lower() == "tools":
                console.print("\n[cyan]📋 Available tools:[/cyan]")
                for tool_name in sorted(TOOLS_REGISTRY.keys()):
                    tool = TOOLS_REGISTRY[tool_name]
                    console.print(f"  • {tool_name}: {tool.description}")
                console.print()
                continue
            elif input_text.lower() == "model":
                console.print("\n[green]🤖 Current model configuration:[/green]")
                console.print(f"  Model: {model_config['model']}")
                console.print(f"  API endpoint: {model_config['base_url']}")
                console.print(f"  API key: {'✓ Set' if model_config['api_key'] else '✗ Missing'}")
                console.print()
                continue

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