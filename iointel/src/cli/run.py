import asyncio
from iointel import Agent
from iointel import AsyncMemory

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
CONVERSATION_ID = "test_conversation_1"

runner = Agent(
    name="Sol",
    instructions="you love to chat and help ppl",
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key="io-v1-c1af24f4f007d634ec37506eb6b86f70129790e0668d045c",
    base_url="https://api.intelligence-dev.io.solutions/api/v1",
    memory=memory,
)


async def main():
    print("Starting chat (type 'exit' to quit)...")
    while True:
        try:
            input_text = input(">>>>: ")
            if input_text.lower() == "exit":
                break
            await runner.run(input_text, conversation_id=CONVERSATION_ID)
            # print(res)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
