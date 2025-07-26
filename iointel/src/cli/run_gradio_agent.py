import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from dotenv import load_dotenv
from iointel import Agent, register_tool, AsyncMemory
from iointel.src.RL.example_tools import (
    gradio_dynamic_ui,
)
from iointel.src.agent_methods.tools.context_tree import ContextTree
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env

load_dotenv("creds.env")

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID
TREE_CONVERSATION_ID = "test_context_tree_02"
AGENT_CONVERSATION_ID = "test_agent_01"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # add your favorite model here
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# tree_agent = get_tree_agent(
#     id_length=7,
#     model_name=os.getenv("MODEL_NAME", "gpt-4o"),
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_API_BASE,
#     memory=memory,
#     conversation_id=TREE_CONVERSATION_ID,
# )

tree = ContextTree(
    id_length=7,
)


# --- REGISTERABLE GENERAL GRADIO PLOTTING TOOL ---
@register_tool
def gradio_plot(
    expression: str = "x**2 - 1",
    x_min: float = -10,
    x_max: float = 10,
    num_points: int = 200,
) -> str:
    """
    Plot a mathematical expression as a function of x using matplotlib and return an HTML <img>.
    expression should be a valid python expression, so use ** for exponentiation, etc.
    """
    x = np.linspace(x_min, x_max, num_points)
    try:
        y = eval(expression, {"x": x, "np": np})
    except Exception as e:
        return f"Error evaluating expression: {e}"
    plt.figure()
    plt.plot(x, y)
    plt.title(f"Plot of: {expression}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}"/>'


# Set up memory
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")
asyncio.run(memory.init_models())


def main():
    # Load all available tools based on environment credentials
    print("Loading tools based on available credentials...")
    available_tools = load_tools_from_env("creds.env")
    
    # Add example tools that aren't already registered
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    example_tools = []
    
    # Only add tools that aren't already in the registry
    if 'gradio_plot' not in TOOLS_REGISTRY:
        example_tools.append(gradio_plot)
    if 'gradio_dynamic_ui' not in TOOLS_REGISTRY:
        example_tools.append(gradio_dynamic_ui)
    
    # Combine all tools (available_tools are already tool names, example_tools are callables)
    all_tools = available_tools + example_tools
    
    print(f"Loaded {len(available_tools)} credential-based tools")
    print(f"Added {len(example_tools)} example/gradio tools")
    print(f"Total tools available: {len(all_tools)}")
    
    # Debug: Show what tools the agent will actually get
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    print(f"All tools in registry: {len(TOOLS_REGISTRY)} tools")
    print(f"Tools agent will actually use: {available_tools}")

    agent = Agent(
        name="IO Solar System",
        instructions="""You are a helpful assistant with access to plotting and dynamic UI tools, arithmetic, time and weather tools, web scraping, search, financial data, cryptocurrency data, file operations, and more depending on what tools are available.
        You can create interactive plots and UI elements for the gradio interface.
        Always explain your reasoning and use tools as needed.""",
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        memory=memory,
        tools=all_tools,
        show_tool_calls=True,
        tool_pil_layout="horizontal",
        debug=True,
    )
    asyncio.run(
        agent.launch_chat_ui(
            interface_title="IO Intel - Solar System", share=False, conversation_id=AGENT_CONVERSATION_ID
        )
    )


if __name__ == "__main__":
    # Load environment variables
    load_dotenv("creds.env")

    main()
