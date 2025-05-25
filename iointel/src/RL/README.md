# IOIntel RL Environment

A reinforcement learning environment for training and evaluating goal-seeking agents. This system allows agents to learn optimal strategies for completing complex tasks through trial and error, with feedback from an oracle.

## Core Components

1. **Environment**: Manages the state, actions, and rewards for the agent
2. **Oracle**: Evaluates agent responses against ground truth
3. **Agent**: The learning agent that uses tools to complete tasks
4. **Tools**: Interface for agent actions (e.g., database queries, API calls)
5. **Critic**: Evaluates agent performance and provides feedback
6. **Task Manager**: Handles task definitions, ground truth, and evaluation criteria

## Directory Structure
```
rl_env/
├── environment.py      # Main RL environment implementation
├── oracle.py          # Oracle for ground truth evaluation
├── agent.py           # RL agent implementation
├── tools/             # Tool implementations
│   ├── base.py        # Base tool interface
│   └── splunk.py      # Example Splunk tool
├── critic.py          # Performance evaluation
├── task_manager.py    # Task and ground truth management
└── utils/             # Utility functions
    ├── metrics.py     # Performance metrics
    └── logging.py     # Experiment logging
``` 

# RL Module

## Agentic Task Generation Example

The RL module supports agentic, LLM-driven task generation. You can generate a curriculum of tasks for any set of tools—no hardcoding required!

### Example: Generate Tasks for Arithmetic Tools

```python
import asyncio
from iointel.src.RL.task_manager import TaskManager
from iointel.src.RL.example_tools import add, subtract, multiply, divide, get_weather

async def main():
    # Create the TaskManager (loads API key from creds.env)
    task_manager = TaskManager(model="gpt-4o")
    tools = [add, subtract, multiply, divide, get_weather]
    # Generate 5 agentic tasks
    tasks = await task_manager.generate_tasks(tools, num_tasks=5)
    for task in tasks:
        print(task)

if __name__ == "__main__":
    asyncio.run(main())
```

**What happens:**
- The TaskManager prompts the LLM with your tool signatures and docstrings.
- The LLM generates a set of diverse, Pydantic `Task` objects tailored to your tools.
- You can use these tasks for RL training, evaluation, or curriculum learning.

**Tip:** Add new tools and the LLM will invent new tasks for them—no code changes needed! 