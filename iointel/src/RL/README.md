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