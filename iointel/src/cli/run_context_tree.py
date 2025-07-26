import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from iointel.src.memory import AsyncMemory
from iointel.src.agent_methods.tools.context_tree import get_tree_agent

load_dotenv("creds.env")

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_context_tree_01"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # add your favorite model here
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

agent = get_tree_agent(
    id_length=7,
    model_name="gpt-4o",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    memory=memory,
    conversation_id=CONVERSATION_ID,
)


async def main():
    console = Console()
    console.print(
        Panel.fit(
            "Starting enhanced chat with tools (type 'exit' to quit)...",
            title="Tool-Enhanced Chat",
            border_style="blue",
        )
    )

    while True:
        try:
            input_text = input(">>>>: ")
            if input_text.lower() == "exit":
                break

            await agent.run(
                input_text, pretty=True
                )
        except Exception as e:
            print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
