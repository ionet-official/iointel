# IOIntel RL Environment

A reinforcement learning environment for training and evaluating goal-seeking agents. This system allows agents to learn optimal strategies for completing complex tasks through trial and error, with feedback from an oracle and a critic.

## Core Components

1. **RLEnvironment**: Orchestrates the agent, critic, oracle, and task manager for RL episodes
2. **OracleAgent**: Evaluates agent responses against ground truth, with robust handling of vague or missing ground truth
3. **CriticAgent**: Evaluates agent performance, provides a score, better query, metrics, and (optionally) improved instructions
4. **TaskManager**: Handles task definitions, ground truth, and evaluation criteria; supports LLM-driven task generation
5. **Tools**: Interface for agent actions (e.g., arithmetic, weather lookup)
6. **Example Tools**: Arithmetic and weather tools for demonstration

## Directory Structure
```
iointel/src/RL/
├── training.py        # Main RL environment and training loop
├── oracle.py          # OracleAgent for ground truth evaluation
├── critic.py          # CriticAgent for agent performance evaluation
├── task_manager.py    # Task and ground truth management, LLM-driven task generation
├── example_tools.py   # Example tool implementations (add, subtract, multiply, divide, get_weather)
├── README.md          # This file
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

---

## RL Pipeline Overview

- **TaskManager** generates or loads tasks, each with a description, ground truth, required tools, and difficulty.
- **RLEnvironment** runs episodes:
  - Instantiates the agent with instructions and tools
  - Agent attempts the task, using tools as needed
  - **CriticAgent** evaluates the agent's actions and response, providing a score, better query, metrics, and (optionally) improved instructions
  - **OracleAgent** evaluates the agent's response against the ground truth, providing correctness, score, and feedback
  - Optionally, the agent can meta-learn by updating its instructions based on critic feedback

## Critic and Oracle Details

- **CriticAgent**: Returns a `CriticFeedback` object with:
  - `score`: float (0.0 to 1.0)
  - `better_query`: improved query or same if already optimal
  - `metrics`: dict of named metrics (e.g., tool_usage_efficiency, response_accuracy)
  - `new_instructions`: (optional) improved agent instructions for meta-learning
- **OracleAgent**: Returns an `EvaluationResult` with:
  - `correct`: bool
  - `score`: float (0.0 to 1.0)
  - `feedback`: string
  - `details`: dict (matching_fields, missing_fields, incorrect_values, additional_insights)
  - Handles vague or missing ground truth by using the task description as fallback

## Best Practices & Learnings

- **Imports**: Use absolute imports for all RL modules to avoid import errors when running scripts directly.
- **Agent/Model Arguments**: Pass model-specific arguments (like `temperature`) via `model_settings` dict, not as top-level arguments to `Agent` or `OpenAIModel`.
- **Tool Usage**: Use the `ToolUsageResult` class to represent agent actions, combining tool name, arguments, and results.
- **Critic/Oracle Output**: Always access the `.result` field of the output dict when using `await agent.run(...)`.
- **Meta-Learning**: The critic can suggest improved instructions for the agent, enabling meta-learning across episodes.

- **Generate New Tasks**: Use `TaskManager.generate_tasks` with your new tools to create a curriculum.
- **Custom Critic/Oracle**: You can subclass `CriticAgent` or `OracleAgent` to change evaluation logic or prompts.
- **Experiment with Meta-Learning**: Enable `meta_learn_instructions` in `RLEnvironment` to let the agent update its instructions based on critic feedback.

---

For more details, see the code in each module and the docstrings. Or run `python iointel/src/RL/training.py` with your own tools.