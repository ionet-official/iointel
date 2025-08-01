import asyncio
from iointel import Agent
from iointel import AsyncMemory
from rich.console import Console
from rich.panel import Panel
import os
from dotenv import load_dotenv
from iointel.src.agent_methods.tools import firecrawl, searxng, wolfram

# Initialize rich console
console = Console()

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_conversation_5"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


# r2r_client = r2r.R2RClient(
#     base_url=os.getenv("R2R_BASE_URL"), api_key=os.getenv("R2R_API_KEY")
# )
firecrawl = firecrawl.Crawler(api_key=os.getenv("FIRE_CRAWL_API_KEY"))
searxng = searxng.SearxngClient(base_url=os.getenv("SEARXNG_URL"))
wolfram = wolfram.Wolfram(api_key=os.getenv("WOLFRAM_API"))

# Create the agent with tools
runner = Agent(
    name="Solar System",
    instructions="""You are a helpful assistant with access to various tools.
    You can perform calculations and check weather information.
    Always explain your reasoning before using a tool.
    After using a tool, explain the result to the user.""",
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    memory=memory,
    tools=[
        #r2r_client.rag_search,
        firecrawl.scrape_url,
        searxng.search,
        searxng.get_urls,
        wolfram.query,
    ],
)


async def main():
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

            # Run the agent with verbose output
            # print(f"Running agent with input: {input_text}", flush=True)
            await runner.run(
                input_text,
                conversation_id=CONVERSATION_ID,
                pretty=True,  # Enable pretty printing
            )
        except Exception as e:
            print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    # Load environment variables from creds.env
    load_dotenv("creds.env")

    asyncio.run(main())
